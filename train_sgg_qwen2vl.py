import os
import json
import random
from typing import Dict, Any, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import HfArgumentParser, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.data.collator.train_collator import MultimodalDataCollator
from PIL import Image

import numpy as np
from tqdm import tqdm

import wandb


# ------------------------------
# Helper Functions
# ------------------------------
def format_bbox_as_special_token(bbox, normalize=True, original_width=1024, original_height=1024):
    """将边界框转换为Qwen2-VL的special token格式"""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        
        if normalize:
            x1_norm = int((x1 / original_width) * 1000)
            y1_norm = int((y1 / original_height) * 1000)
            x2_norm = int((x2 / original_width) * 1000)
            y2_norm = int((y2 / original_height) * 1000)
            
            x1_norm = max(0, min(x1_norm, 999))
            y1_norm = max(0, min(y1_norm, 999))
            x2_norm = max(0, min(x2_norm, 999))
            y2_norm = max(0, min(y2_norm, 999))
            
            x1_norm, x2_norm = min(x1_norm, x2_norm), max(x1_norm, x2_norm)
            y1_norm, y2_norm = min(y1_norm, y2_norm), max(y1_norm, y2_norm)
            
            if x1_norm == x2_norm:
                x2_norm = min(x1_norm + 1, 999)
            if y1_norm == y2_norm:
                y2_norm = min(y1_norm + 1, 999)
            
            return f"<|box_start|>({x1_norm}, {y1_norm}), ({x2_norm}, {y2_norm})<|box_end|>"
    return ""

def format_object_with_ref(object_label):
    """将物体标签包装在对象引用token中"""
    return f"<|object_ref_start|>{object_label}<|object_ref_end|>"


# image token placeholder
VLM_IMAGE_TOKENS = {"QWEN2_VL": "<|image_pad|>"}  # 注意这里必须是 <|image_pad|>
QWEN2_VL = "QWEN2_VL"



# ------------------------------
# Dataset
# ------------------------------
class SGGContrastiveDataset(Dataset):
    def __init__(
        self, 
        json_path: str, 
        image_dir: str, 
        relation_vocabulary: Optional[List[str]] = None, 
        num_negatives: int = 12,
        topk_nearest: Optional[dict] = None,
        topk_nearest_file: Optional[str] = None
    ):
        # 读取样本
        with open(json_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                self.samples = json.loads(content)
            else:
                self.samples = [json.loads(l) for l in content.splitlines() if l.strip()]

        self.image_dir = image_dir
        self.num_negatives = num_negatives

        # 构建 predicate vocab
        if relation_vocabulary is None:
            rels = set()
            for s in self.samples:
                if 'predicate' in s and s['predicate']:
                    rels.add(s['predicate'])
            self.vocab = sorted(list(rels))
        else:
            self.vocab = relation_vocabulary

        if len(self.vocab) == 0:
            raise ValueError("Empty relation vocabulary")

        # 处理 top-k nearest
        if topk_nearest is not None:
            self.topk_nearest = topk_nearest
        elif topk_nearest_file is not None:
            with open(topk_nearest_file, 'r') as f:
                data = json.load(f)
                self.topk_nearest = {k: v["neighbors"] for k, v in data.items()}
        else:
            self.topk_nearest = {}

    def __len__(self):
        return len(self.samples)

    def _full_image_path(self, path: Optional[str]):
        """Return the absolute path for the image or None if path is missing.

        Accepts None and returns None so callers can handle missing images explicitly.
        """
        if not path:
            return None
        return path if os.path.isabs(path) else os.path.join(self.image_dir, path)

    def _make_image_field(self, path: Optional[str]):
        full = self._full_image_path(path)
        # If no path, return a placeholder image field where path is None.
        return {'resolutions': [None], 'paths': [full], 'bytes': [None]}

    def _get_image_dimensions(self, image_path: Optional[str]):
        try:
            if image_path is None:
                # Missing image, return default fallback dimensions
                return (1024, 1024)
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            print(f"Warning: Failed to get image dimensions for {image_path}: {e}")
            return (1024, 1024)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image_path = s.get('image_path') or s.get('img_path') or s.get('image')
        predicate = s.get('predicate') or s.get('relation') or "related"
        subj = s.get('subject', {})
        obj = s.get('object', {})
        bbox1 = subj.get('bbox') or s.get('bbox1')
        bbox2 = obj.get('bbox') or s.get('bbox2')
        subj_name = subj.get('class_name', 'objectA')
        obj_name = obj.get('class_name', 'objectB')

        # Check for missing image and handle accordingly (raise or return placeholder)
        if image_path is None:
            # 可以选择抛出错误或返回一个占位样本（下面演示抛错）
            raise ValueError(f"Missing image path for sample index {idx}: {s.get('id', idx)}")

        # 图像信息
        full_image_path = self._full_image_path(image_path)
        original_width, original_height = self._get_image_dimensions(full_image_path)

        # 格式化 token
        subj_bbox_token = format_bbox_as_special_token(bbox1, True, original_width, original_height)
        obj_bbox_token = format_bbox_as_special_token(bbox2, True, original_width, original_height)
        subj_ref = format_object_with_ref(subj_name)
        obj_ref = format_object_with_ref(obj_name)

        query_text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]} In the given image, the subject {subj_ref} is located at {subj_bbox_token}, the object {obj_ref} is located at {obj_bbox_token}. Please return the predicate relationship between the subject and the object."
        pos_text = f"The subject is {predicate} the object."

        # 负样本生成（hard nearest + random）
        neg_candidates = [r for r in self.vocab if r != predicate]

        hard_negatives = []
        if self.topk_nearest:
            hard_negatives = [r for r in self.topk_nearest.get(predicate, []) if r != predicate]

        remaining = max(self.num_negatives - len(hard_negatives), 0)
        if neg_candidates:
            random_negatives = random.sample(neg_candidates, min(remaining, len(neg_candidates)))
        else:
            random_negatives = []

        final_negatives = hard_negatives + random_negatives
        neg_texts = [f"The subject is {r} the object." for r in final_negatives]

        # image field
        img_field = self._make_image_field(image_path)
        query_image = img_field
        pos_image = img_field
        neg_images = [img_field] * len(neg_texts)

        return {
            'query_text': query_text,
            'query_image': query_image,
            'pos_text': pos_text,
            'pos_image': pos_image,
            'neg_text': neg_texts,
            'neg_image': neg_images,
            'global_dataset_name': s.get('dataset_name', 'vg')
        }



# ------------------------------
# Utils
# ------------------------------
def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(x, device) for x in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


def evaluate(model, val_loader, device, return_scores=False):
    """
    评估模型平均 InfoNCE loss，同时可选择返回最后一个 batch 的 scores。
    """
    model.eval()
    losses = []
    last_scores = None

    with torch.no_grad():
        for batch in val_loader:
            qry_inputs, pos_inputs, neg_inputs = batch  # collator 返回三部分

            qry_inputs = batch_to_device(qry_inputs, device)
            pos_inputs = batch_to_device(pos_inputs, device)
            neg_inputs = batch_to_device(neg_inputs, device)

            # forward 返回 dict 包含 loss 和 scores
            out = model(qry=qry_inputs, tgt=pos_inputs, neg=neg_inputs)
            loss_tensor = out["loss"]
            scores = out.get("scores", None)

            losses.append(loss_tensor.item())
            last_scores = scores  # 记录最后一个 batch 的 scores

    model.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    if return_scores:
        return mean_loss, last_scores
    return mean_loss



# ------------------------------
# Train Loop
# ------------------------------
def train_loop(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # Determine distributed / multi-GPU setup
    use_ddp = getattr(training_args, 'use_ddp', False) or int(os.environ.get("WORLD_SIZE", "1")) > 1
    dist_initialized = False
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if use_ddp:
        # Initialize distributed process group
        backend = getattr(training_args, 'ddp_backend', 'nccl')
        dist.init_process_group(backend=backend)
        dist_initialized = True
        rank = dist.get_rank()
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading processor...")
    processor = load_processor(model_args)

    print("Building model...")
    model = MMEBModel.build(model_args)
    

    # 应用 LoRA
    if model_args.lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(","),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        print("✅ LoRA applied")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    # Move model to device after applying LoRA (correct ordering)
    model = model.to(device)
    model.train()

    # -----------------------------
    # Initialize W&B
    # -----------------------------
    # Wrap model for multi-GPU
    if dist_initialized:
        # DistributedDataParallel: wrap the already-device-placed model
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        is_main_process = (dist.get_rank() == 0)
    elif torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # DataParallel fallback for single-node multi-GPU
        model = nn.DataParallel(model)
        is_main_process = True
    else:
        is_main_process = True

    # Initialize W&B only on main process to avoid duplicate logs
    if is_main_process:
        wandb.init(
            project="sgg_qwen2vl",
            name=os.path.basename(training_args.output_dir.rstrip("/")),
            config={**vars(model_args), **vars(data_args), **vars(training_args)},
            mode="offline"  # <-- 离线模式
        )

    # 准备数据
    dataset = SGGContrastiveDataset(
        json_path=data_args.dataset_json,             # JSON 数据集路径
        image_dir=data_args.image_dir,               # 图像目录
        num_negatives=data_args.num_negatives,       # 每个样本负样本总数
        topk_nearest_file=getattr(data_args, 'topk_nearest_file', None)  # 可选 topk nearest JSON 文件
    )
    collator = MultimodalDataCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        batch_size=training_args.per_device_train_batch_size
    )

    # Use DistributedSampler when DDP is enabled
    if dist_initialized:
        train_sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,  # 可根据需要调整
            pin_memory=True if torch.cuda.is_available() else False
        )

    # ---------- evaluation dataset/loader (optional) ----------
    # Use the eval path supplied in DataArguments. The default should be set in `src/arguments.py`.
    val_loader = None
    eval_json = getattr(data_args, 'eval_dataset_json', None)
    if eval_json:
        try:
            val_dataset = SGGContrastiveDataset(
                eval_json,
                data_args.image_dir,
                num_negatives=data_args.num_negatives
            )
            eval_batch_size = getattr(training_args, 'per_device_eval_batch_size', training_args.per_device_train_batch_size)
            if dist_initialized:
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    sampler=val_sampler,
                    collate_fn=collator,
                    num_workers=0,
                    pin_memory=True if torch.cuda.is_available() else False,
                )
            else:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=0,
                    pin_memory=True if torch.cuda.is_available() else False
                )
            print(f"✅ Eval dataset loaded: {len(val_dataset)} samples from {eval_json}")
        except Exception as e:
            val_loader = None
            print(f"⚠️  Failed to load eval dataset from {eval_json}: {e}")

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay if hasattr(training_args, 'weight_decay') else 0.01
    )

    # 计算总步数
    num_update_steps_per_epoch = len(dataloader) // training_args.gradient_accumulation_steps
    total_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch
    warmup_steps = int(total_steps * (training_args.warmup_ratio if hasattr(training_args, 'warmup_ratio') else 0.1))

    # Cosine 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    global_step = 0
    # Loss 记录
    loss_history = {
        'steps': [],
        'losses': [],
        'epoch_losses': [],
        'eval_losses': [],  # eval loss at each epoch
        'eval_losses_steps': [],  # eval loss at each eval step
        'eval_steps': []  # step indices for eval loss
    }
    best_loss = float('inf')
    best_eval_loss = float('inf')


    print("WORLD_SIZE =", os.environ.get("WORLD_SIZE"))
    print("LOCAL_RANK =", os.environ.get("LOCAL_RANK"))
    print("DDP =", use_ddp)


    # 训练循环
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        if dist_initialized:
            try:
                dataloader.sampler.set_epoch(epoch)
            except Exception:
                pass
        
        for batch_idx, (qry_inputs, pos_inputs, neg_inputs) in enumerate(dataloader):
            qry_inputs = batch_to_device(qry_inputs, device)
            pos_inputs = batch_to_device(pos_inputs, device)
            neg_inputs = batch_to_device(neg_inputs, device)
            # 前向传播
            loss = model(qry=qry_inputs, tgt=pos_inputs, neg=neg_inputs)
            loss_tensor = loss["loss"] if isinstance(loss, dict) else loss
            
            # 梯度累积
            loss_tensor = loss_tensor / training_args.gradient_accumulation_steps
            loss_tensor.backward()

            # 记录 loss
            epoch_losses.append(loss_tensor.item() * training_args.gradient_accumulation_steps)

            # 更新参数
            if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
                # 梯度裁剪(可选)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Log to W&B
                current_lr = scheduler.get_last_lr()[0]
                step_loss = loss_tensor.item() * training_args.gradient_accumulation_steps
                if is_main_process:
                    wandb.log({
                        "train/loss_step": step_loss,
                        "lr": current_lr,
                        "epoch": epoch+1,
                        "global_step": global_step
                    }, step=global_step)
                    if global_step % 100 == 0:
                        print(f"[Step {global_step}] Epoch {epoch+1} | Loss: {step_loss:.4f} | LR: {current_lr:.2e}")

                global_step += 1

                # 定期 evaluation（如果配置了 eval_steps 并且成功加载了 val_loader）
                if hasattr(training_args, 'eval_steps') and getattr(training_args, 'eval_steps') and val_loader is not None:
                    if global_step % int(training_args.eval_steps) == 0:
                        # Only run evaluation and saving on the main process
                        if is_main_process:
                            val_loss = evaluate(model, val_loader, device)
                            wandb.log({"eval/loss": val_loss, "step": global_step})
                            print(f"[Eval Step {global_step}] Validation Loss: {val_loss:.4f}")

                            # 根据 eval loss 保存最佳模型（可选）
                            if val_loss < best_eval_loss:
                                best_eval_loss = val_loss
                                best_dir_eval = os.path.join(training_args.output_dir, "best_model_eval")
                                os.makedirs(best_dir_eval, exist_ok=True)
                                model_to_save = model.module if hasattr(model, 'module') else model
                                model_to_save.save(best_dir_eval)
                                print(f"🏆 New best eval model saved at step {global_step} (val_loss: {best_eval_loss:.4f})")
                           
        # Epoch 结束
        avg_epoch_loss = np.mean(epoch_losses)
        if is_main_process:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch+1})
            print(f"📊 Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f} [W&B logged]\n")

        # Evaluate at the end of each epoch and track eval loss (only main process)
        if val_loader is not None and is_main_process:
            val_loss_epoch = evaluate(model, val_loader, device)
            wandb.log({"eval/epoch_loss": val_loss_epoch, "epoch": epoch+1})
            print(f"📊 Epoch {epoch+1} Validation Loss: {val_loss_epoch:.4f} [W&B logged]")

        if is_main_process:
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.6f}")
            print(f"  Min Loss: {min(epoch_losses):.6f}")
            print(f"  Max Loss: {max(epoch_losses):.6f}")

        # 保存 checkpoint (main process only)
        if is_main_process and ((epoch + 1) % training_args.save_steps == 0 or (epoch + 1) == int(training_args.num_train_epochs)):
            save_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save(save_dir)
            print(f"💾 Checkpoint saved to: {save_dir}")

        # 保存最佳模型 (main process only)
        if is_main_process and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            wandb.run.summary["best_train_loss"] = best_loss
            best_dir = os.path.join(training_args.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save(best_dir)
            print(f"🏆 Best model saved — loss: {best_loss:.6f} — {best_dir}")

    # 最终保存 (只在主进程执行)
    if is_main_process:
        final_dir = os.path.join(training_args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save(final_dir)
        wandb.finish()

        print(f"\n{'='*60}")
        print("✅ Training Complete!")
        print(f"{'='*60}")
        print(f"📁 Final model saved to: {final_dir}")
        print(f"🏆 Best model saved to: {os.path.join(training_args.output_dir, 'best_model')}")
        print(f"🎯 Best loss: {best_loss:.6f}")

    # Cleanup distributed process group
    if dist_initialized:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# ------------------------------
# Main
# ------------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 使用 checkpoint path 作为 model name (离线加载)
    if model_args.checkpoint_path is not None:
        model_args.model_name = model_args.checkpoint_path

    train_loop(model_args, data_args, training_args)


if __name__ == "__main__":
    main()