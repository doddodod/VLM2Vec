#!/usr/bin/env python
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
import lora


# ------------------------------
# Helper Functions
# ------------------------------
def format_bbox_as_special_token(bbox, normalize=True, original_width=1024, original_height=1024):
    """将边界框转换为Qwen2-VL的special token格式"""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        
        if normalize:
            x1_norm = int((x1 / original_width) * 1000) if original_width else 0
            y1_norm = int((y1 / original_height) * 1000) if original_height else 0
            x2_norm = int((x2 / original_width) * 1000) if original_width else 0
            y2_norm = int((y2 / original_height) * 1000) if original_height else 0
            
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


def enable_gc_for_encoder(encoder, _seen=None):
    if _seen is None:
        _seen = set()

    unwrapped = try_unwrap_peft(encoder)

    uid = id(unwrapped)
    if uid in _seen:
        return
    _seen.add(uid)

    # Skip LoRA leaf modules
    if "lora" in unwrapped.__class__.__name__.lower():
        return

    # Disable KV cache
    if hasattr(unwrapped, "config"):
        try:
            unwrapped.config.use_cache = False
        except Exception:
            pass

    # Enable checkpointing
    if hasattr(unwrapped, "gradient_checkpointing_enable"):
        try:
            unwrapped.gradient_checkpointing_enable()
            print(f"🔥 gradient_checkpointing_enable() applied on {unwrapped.__class__.__name__}")
        except Exception as e:
            print(f"⚠ Failed to apply gradient_checkpointing_enable() on {type(unwrapped).__name__}: {e}")

    if hasattr(unwrapped, "enable_input_require_grads"):
        try:
            unwrapped.enable_input_require_grads()
            print(f"🔥 enable_input_require_grads() applied on {unwrapped.__class__.__name__}")
        except Exception as e:
            print(f"⚠ Failed to apply enable_input_require_grads() on {type(unwrapped).__name__}: {e}")

    # Recurse into backbone submodules
    for name in ["model", "transformer", "language_model", "vision_model", "vision_tower"]:
        if hasattr(unwrapped, name):
            sub = getattr(unwrapped, name)
            print(f"➡ enabling GC for submodule: {type(unwrapped).__name__}.{name}")
            enable_gc_for_encoder(sub, _seen=_seen)

# ---------- 安全检查并报告 encoder 各子模块的 gradient_checkpointing 状态 ----------
def safe_get_submodule(root, path):
    """按 path（'a.b.c'）安全地遍历属性，任意一步失败返回 None。"""
    cur = root
    for part in path.split('.'):
        try:
            cur = getattr(cur, part)
        except Exception:
            return None
    return cur

def try_unwrap_peft(m):
    """如果是 PEFT wrapper，尽量拿到 base model；否则返回原对象。"""
    # PeftModel 可能把真实模型放在 base_model 或 model 或 underlying_model
    for attr in ("base_model", "model", "_wrapped_model"):
        if hasattr(m, attr):
            try:
                inner = getattr(m, attr)
                if inner is not None:
                    return inner
            except Exception:
                pass
    return m

def report_gradient_checkpointing(encoder):
    enc = try_unwrap_peft(encoder)
    print(f"→ Reporting gradient_checkpointing for top encoder type: {type(enc).__name__}")

    # 常见的候选路径（按 Qwen2-VL 及类似模型结构列）
    candidate_paths = [
        "",  # top-level encoder itself
        "model",
        "model.vision_tower",
        "model.vision_tower.model",
        "model.layers",
        "model.layers.0",
        "model.layers.0.self_attn",
        "model.norm",
        "vision_tower",
        "language_model",
        "transformer",
    ]

    seen = set()
    for p in candidate_paths:
        target = enc if p == "" else safe_get_submodule(enc, p)
        if target is None:
            continue
        tname = p if p != "" else "<encoder>"
        # 避免重复打印同一对象
        obj_id = id(target)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        # safe read gradient_checkpointing attribute
        gc_val = None
        try:
            gc_val = getattr(target, "gradient_checkpointing", None)
        except Exception:
            gc_val = None
        print(f"{tname:30} | type={type(target).__name__:30} | gradient_checkpointing={gc_val}")

    # 最后，扫描 encoder.named_modules 中有该属性的模块（更全面）
    try:
        count = 0
        for name, sub in enc.named_modules():
            # 过滤非常短的子模块名让输出可控
            if len(name) == 0 or name.count('.') > 3:
                continue
            if hasattr(sub, "gradient_checkpointing"):
                try:
                    val = getattr(sub, "gradient_checkpointing")
                except Exception:
                    val = None
                print(f"named_module: {name:40} | {type(sub).__name__:30} | gradient_checkpointing={val}")
                count += 1
                if count >= 30:
                    break
        if count == 0:
            print("No named submodules with explicit `gradient_checkpointing` attribute found in a quick scan.")
    except Exception as e:
        print("Warning: scanning named_modules failed:", e)


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
                # 支持逐行 json
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
                # 期望每个 key 下有 "neighbors" 列表
                self.topk_nearest = {k: v.get("neighbors", []) for k, v in data.items()}
        else:
            self.topk_nearest = {}

    def __len__(self):
        return len(self.samples)

    def _full_image_path(self, path: Optional[str]):
        """Return the absolute path for the image or None if path is missing."""
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
            raise ValueError(f"Missing image path for sample index {idx}: {s.get('id', idx)}")

        # 图像信息
        full_image_path = self._full_image_path(image_path)
        original_width, original_height = self._get_image_dimensions(full_image_path)

        # 格式化 token
        subj_bbox_token = format_bbox_as_special_token(bbox1, True, original_width, original_height)
        obj_bbox_token = format_bbox_as_special_token(bbox2, True, original_width, original_height)
        subj_ref = format_object_with_ref(subj_name)
        obj_ref = format_object_with_ref(obj_name)

        query_text = f"{VLM_IMAGE_TOKENS.get(QWEN2_VL, '<|image_pad|>')} In the given image, the subject {subj_ref} is located at {subj_bbox_token}, the object {obj_ref} is located at {obj_bbox_token}. Please return the predicate relationship between the subject and the object."
        pos_text = f"The subject is {predicate} the object."

        # 负样本生成（hard nearest + random）
        neg_candidates = [r for r in self.vocab if r != predicate]

        hard_negatives = []
        if self.topk_nearest:
            hard_negatives = [r for r in self.topk_nearest.get(predicate, []) if r != predicate]

        remaining = max(self.num_negatives - len(hard_negatives), 0)
        if neg_candidates:
            # 如果候选数不足，sample 会处理
            k = min(remaining, len(neg_candidates))
            random_negatives = random.sample(neg_candidates, k) if k > 0 else []
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



def evaluate(model, val_loader, device, is_distributed=False, return_scores=False):
    """
    增强版本：支持更多诊断信息。
    """
    model_to_eval = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    model_to_eval.eval()
    
    all_losses = []
    last_scores = None
    batch_count = 0

    with torch.no_grad():
        for batch in val_loader:
            qry_inputs, pos_inputs, neg_inputs = batch

            qry_inputs = batch_to_device(qry_inputs, device)
            pos_inputs = batch_to_device(pos_inputs, device)
            neg_inputs = batch_to_device(neg_inputs, device)

            out = model_to_eval(qry=qry_inputs, tgt=pos_inputs, neg=neg_inputs)
            loss_tensor = out["loss"] if isinstance(out, dict) else out
            scores = out.get("scores", None) if isinstance(out, dict) else None

            all_losses.append(loss_tensor.detach().cpu())
            last_scores = scores
            batch_count += 1

    # 计算本进程的平均 loss
    if all_losses:
        local_losses_tensor = torch.stack(all_losses)  # [num_batches]
        local_mean_loss = local_losses_tensor.mean()
    else:
        local_mean_loss = torch.tensor(0.0, device=device)

    # ===== DDP 同步 =====
    if is_distributed:
        # 方法1：汇聚 loss 张量本身
        loss_tensor = local_mean_loss.to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        mean_loss = float(loss_tensor.item() / world_size)
        
        # 仅在 rank 0 打印日志
        if dist.get_rank() == 0:
            print(f"[Evaluate] Mean loss across {world_size} processes: {mean_loss:.6f}")
    else:
        mean_loss = float(local_mean_loss.item())

    model_to_eval.train()
    
    if return_scores:
        return mean_loss, last_scores
    return mean_loss


def get_model_parameters_for_optimizer(model):
    """获取优化器所需的参数，正确处理 DDP/DataParallel wrapper"""
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        return [p for p in model.module.parameters() if p.requires_grad]
    else:
        return [p for p in model.parameters() if p.requires_grad]


def clip_grad_norm(model, max_norm=1.0):
    """在 DDP/DataParallel 下正确进行梯度裁剪"""
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def set_seed(seed: int, deterministic: bool = True):
    """设置全局随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 为了可复现，禁用某些非确定性算法（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model_safe(model_obj, out_dir):
    """兼容多种保存 API"""
    os.makedirs(out_dir, exist_ok=True)
    if hasattr(model_obj, "save_pretrained"):
        model_obj.save_pretrained(out_dir)
    elif hasattr(model_obj, "save"):
        # 自定义 model.save
        model_obj.save(out_dir)
    else:
        torch.save(model_obj.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))


# ------------------------------
# Train Loop
# ------------------------------
def train_loop(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # Decide whether to use DDP based on env (torchrun) or args
    # Prefer environment-driven DDP (torchrun sets WORLD_SIZE, LOCAL_RANK)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = getattr(training_args, 'use_ddp', False) or world_size_env > 1
    dist_initialized = False
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize distributed process group if appropriate
    if use_ddp:
        backend = getattr(training_args, 'ddp_backend', 'nccl')
        # 必须先 set device，然后 init_process_group
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception:
                pass
        try:
            # init uses env vars set by torchrun
            if dist.is_available() and dist.is_initialized():
                # Already initialized elsewhere (e.g., calling program); use it
                dist_initialized = True
                rank = dist.get_rank()
                if rank == 0:
                    print("⚠️  Warning: dist.process_group already initialized, skipping init_process_group")
            else:
                dist.init_process_group(backend=backend)
                dist_initialized = True
                rank = dist.get_rank()
        except Exception as e:
            # If the error indicates double initialization, try to recover by checking is_initialized
            if dist.is_available() and dist.is_initialized():
                dist_initialized = True
                rank = dist.get_rank()
                if rank == 0:
                    print(f"⚠️  DDP init raised an exception but process group is initialized: {e}")
            elif use_ddp:  # 如果明确要求 DDP 但失败了，应该失败而不是降级
                raise RuntimeError(f"Failed to initialize DDP: {e}")
            else:
                print("⚠️  Warning: dist.init_process_group skipped (not DDP mode)")
                dist_initialized = False

    # Device for this process
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    is_main_process = (rank == 0)

    # 设置随机数种子
    seed = getattr(training_args, 'seed', 42)
    deterministic = getattr(training_args, 'deterministic', True)
    set_seed(seed, deterministic=deterministic)
    if is_main_process:
        print(f"🌱 Random seed set to: {seed} (deterministic={deterministic})")

    if is_main_process:
        print(f"Using device: {device} | world_size: {dist.get_world_size() if dist_initialized else 1} | local_rank: {local_rank}")

    # 加载 processor & model（各进程本地加载）
    if is_main_process:
        print("Loading processor...")
    processor = load_processor(model_args)

    if is_main_process:
        print("Building model...")
        
    model = MMEBModel.build(model_args)

    if training_args.gradient_checkpointing:
        print("\n============================")
        print("🔥 Enabling REAL gradient checkpointing...")
        print("============================\n")

        # 你真正的模型在 model.encoder 里
        enable_gc_for_encoder(model.encoder)

        print("\n============================")
        print("🔥 Gradient checkpointing ENABLED")
        print("============================\n")


    # Move model to device
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(model)
        try:
            report_gradient_checkpointing(model.encoder)
        except Exception as e:
            print("report_gradient_checkpointing raised exception:", e)
    
    model = model.to(device)
    model.train()

    # Wrap model for DDP/DataParallel
    if dist_initialized:
        find_unused = getattr(training_args, "find_unused_parameters", True)
        # device_ids must be set when using single-process single-device per rank (torchrun)
        device_ids = [local_rank] if torch.cuda.is_available() else None
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused,
        )
    elif torch.cuda.device_count() > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model)

    # If using DDP + gradient checkpointing, it's often necessary to set a static
    # graph to avoid multiple reentrant backward passes for the same parameters.
    # We call _set_static_graph() on the underlying module if available.
    try:
        if dist_initialized and getattr(training_args, 'gradient_checkpointing', False):
            base_mod = model.module if hasattr(model, 'module') else model
            base_mod = try_unwrap_peft(base_mod)
            if hasattr(base_mod, '_set_static_graph'):
                try:
                    base_mod._set_static_graph()
                    if is_main_process:
                        print("🔒 Set static graph on base module to avoid DDP reentrant backward issues")
                except Exception as e:
                    if is_main_process:
                        print(f"⚠️ _set_static_graph() failed: {e}")
    except Exception:
        pass

    # Initialize W&B only on main process to avoid duplicate logs
    if is_main_process and getattr(training_args, "use_wandb", True):
        wandb.init(
            project=getattr(training_args, "wandb_project", "sgg_qwen2vl"),
            name=os.path.basename(training_args.output_dir.rstrip("/")),
            config={**vars(model_args), **vars(data_args), **vars(training_args)},
            mode=getattr(training_args, "wandb_mode", "offline")
        )

    # 准备数据集与 collator（每个进程本地构造）
    dataset = SGGContrastiveDataset(
        json_path=data_args.dataset_json,             # JSON 数据集路径
        image_dir=data_args.image_dir,               # 图像目录
        num_negatives=getattr(data_args, "num_negatives", 12),
        topk_nearest_file=getattr(data_args, 'topk_nearest_file', None)
    )
    collator = MultimodalDataCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        batch_size=training_args.per_device_train_batch_size
    )

    # DataLoader 配置
    num_workers = getattr(training_args, 'dataloader_num_workers', 4)
    pin_memory = torch.cuda.is_available() and num_workers > 0

    if dist_initialized:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=seed
        )
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # 可选 eval dataset
    val_loader = None
    eval_json = getattr(data_args, 'eval_dataset_json', None)
    if eval_json:
        try:
            val_dataset = SGGContrastiveDataset(
                eval_json,
                data_args.image_dir,
                num_negatives=getattr(data_args, "num_negatives", 12)
            )
            eval_batch_size = getattr(training_args, 'per_device_eval_batch_size', training_args.per_device_train_batch_size)
            if dist_initialized:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=rank,
                    shuffle=False,
                    seed=seed,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    sampler=val_sampler,
                    collate_fn=collator,
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
            else:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
            if is_main_process:
                print(f"✅ Eval dataset loaded: {len(val_dataset)} samples from {eval_json}")
        except Exception as e:
            val_loader = None
            if is_main_process:
                print(f"⚠️  Failed to load eval dataset from {eval_json}: {e}")

    # 获取优化器参数（在 wrap 之后，确保参数正确）
    optimizer_params = get_model_parameters_for_optimizer(model)
    if len(optimizer_params) == 0:
        raise ValueError("⚠️ No trainable parameters found! Check your model and LoRA config.")

    # 优化器
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=training_args.learning_rate,
        weight_decay=getattr(training_args, 'weight_decay', 0.01)
    )

    # ===== 修复：正确计算 total steps =====
    num_batches_per_epoch = len(dataloader)  # 这已经是当前进程的 batch 数

    # 在 DDP 下同步所有进程的 batch 数量（确保一致性）
    if dist_initialized:
        num_batches_tensor = torch.tensor(num_batches_per_epoch, dtype=torch.long, device=device)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.MIN)
        num_batches_per_epoch = int(num_batches_tensor.item())

    # 计算 update steps
    num_update_steps_per_epoch = max(1, num_batches_per_epoch // training_args.gradient_accumulation_steps)
    total_steps = max(1, int(training_args.num_train_epochs) * num_update_steps_per_epoch)
    warmup_steps = max(0, int(total_steps * getattr(training_args, 'warmup_ratio', 0.1)))

    if is_main_process:
        print(f"📊 Training config:")
        print(f"  - Batches per epoch (per device): {num_batches_per_epoch}")
        print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"  - Update steps per epoch: {num_update_steps_per_epoch}")
        print(f"  - Total epochs: {int(training_args.num_train_epochs)}")
        print(f"  - Total update steps: {total_steps}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Distributed: {dist_initialized}")
        print(f"  - World size: {dist.get_world_size() if dist_initialized else 1}")

    # 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 初始化跟踪变量
    global_step = 0
    best_loss = float('inf')
    best_eval_loss = float('inf')

    # 调试信息
    if is_main_process:
        print(f"\n🔧 Environment Info:")
        print(f"  - WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}")
        print(f"  - LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')}")
        print(f"  - RANK: {os.environ.get('RANK', 'N/A')}")
        print(f"  - Use DDP: {use_ddp}")
        print(f"  - Device: {device}\n")

    # --- main training loop ---
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        # 在 DDP 下每个 epoch 需要设置 sampler 的 epoch，以改变 shuffle seed
        if dist_initialized:
            try:
                dataloader.sampler.set_epoch(epoch)
            except Exception:
                pass

        # 可以在主进程使用 tqdm
        iterator = enumerate(dataloader)
        if is_main_process and getattr(training_args, "show_progress_bar", False):
            iterator = enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}"))

        for batch_idx, batch in iterator:
            # 假定 collator 返回 (qry_inputs, pos_inputs, neg_inputs)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                qry_inputs, pos_inputs, neg_inputs = batch
            else:
                # 如果 collator 返回 dict 或其它结构，按你的 collator 修改这里
                raise ValueError("Expected collator to return (qry_inputs, pos_inputs, neg_inputs)")

            qry_inputs = batch_to_device(qry_inputs, device)
            pos_inputs = batch_to_device(pos_inputs, device)
            neg_inputs = batch_to_device(neg_inputs, device)

            # forward
            out = model(qry=qry_inputs, tgt=pos_inputs, neg=neg_inputs)
            loss_tensor = out["loss"] if isinstance(out, dict) else out
            assert isinstance(loss_tensor, torch.Tensor), f"Expected Tensor, got {type(loss_tensor)}"

            original_loss = loss_tensor.detach().item()
            loss = loss_tensor / training_args.gradient_accumulation_steps
            loss.backward()

            epoch_losses.append(original_loss)

            # 梯度累积步满足才更新参数
            if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                clip_grad_norm(model, max_norm=getattr(training_args, "max_grad_norm", 1.0))

                 # DDP 下的梯度同步屏障（可选但推荐）
                if dist_initialized:
                    dist.barrier()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 更新 global step（在 optimizer.step 后）
                global_step += 1

                # 计算累积后的平均 loss（用于日志）
                accumulated_batch_losses = epoch_losses[-training_args.gradient_accumulation_steps:]
                accumulated_loss_avg = float(np.mean(accumulated_batch_losses))

                # logging（仅主进程打印/上传）
                current_lr = scheduler.get_last_lr()[0]
                if is_main_process and getattr(training_args, "use_wandb", True):
                    wandb.log({
                        "train/loss_step": accumulated_loss_avg,  # ← 修复：使用累积后的平均 loss
                        "train/loss_batch": original_loss,  # ← 额外：原始 batch loss
                        "lr": current_lr,
                        "epoch": epoch + 1,
                        "global_step": global_step
                    }, step=global_step)
                if is_main_process and global_step % 100 == 0:
                    print(f"[Step {global_step}] Epoch {epoch+1}/{int(training_args.num_train_epochs)} | AccumLoss: {accumulated_loss_avg:.4f} | LR: {current_lr:.2e}")

                # 定期 evaluation
                eval_steps = getattr(training_args, 'eval_steps', None)
                if eval_steps is not None and eval_steps > 0 and val_loader is not None:
                    if global_step % int(training_args.eval_steps) == 0:
                        val_loss = evaluate(model, val_loader, device, is_distributed=dist_initialized)
                        if is_main_process and getattr(training_args, "use_wandb", True):
                            wandb.log({"eval/loss": val_loss}, step=global_step)
                        if is_main_process:
                            print(f"[Eval Step {global_step}] Validation Loss: {val_loss:.4f}")

                        # 根据 eval loss 保存最佳模型（仅主进程保存）
                        if is_main_process and val_loss < best_eval_loss:
                            best_eval_loss = val_loss
                            best_dir_eval = os.path.join(training_args.output_dir, "best_model_eval")
                            save_model_safe(model.module if hasattr(model, 'module') else model, best_dir_eval)
                            print(f"🏆 New best eval model saved at step {global_step} (val_loss: {best_eval_loss:.4f})")

        # epoch 结束
        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # 修复：在 DDP 下同步 epoch loss
        if dist_initialized:
            loss_tensor = torch.tensor(avg_epoch_loss, dtype=torch.float32, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            avg_epoch_loss = float(loss_tensor.item() / world_size)

        if is_main_process and getattr(training_args, "use_wandb", True):
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch+1}, step=global_step)
        if is_main_process:
            print(f"📊 Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f} [W&B logged]\n")

        # epoch 末尾 evaluate
        if val_loader is not None:
            val_loss_epoch = evaluate(model, val_loader, device, is_distributed=dist_initialized)
            if is_main_process and getattr(training_args, "use_wandb", True):
                wandb.log({"eval/epoch_loss": val_loss_epoch, "epoch": epoch+1}, step=global_step)
            if is_main_process:
                print(f"📊 Epoch {epoch+1} Validation Loss: {val_loss_epoch:.4f} [W&B logged]")

         # ===== 保存 checkpoint（修复 save_steps 逻辑）=====
        save_steps = getattr(training_args, "save_steps", None)
        should_save_ckpt = False
        
        if save_steps is None:
            # 默认：只在最后一个 epoch 保存
            should_save_ckpt = ((epoch + 1) == int(training_args.num_train_epochs))
        else:
            # 按指定间隔保存，或最后一个 epoch
            should_save_ckpt = ((epoch + 1) % int(save_steps) == 0 or (epoch + 1) == int(training_args.num_train_epochs))
        
        if is_main_process and should_save_ckpt:
            save_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch+1}")
            save_model_safe(model.module if hasattr(model, 'module') else model, save_dir)
            print(f"💾 Checkpoint saved to: {save_dir}")

        # ===== 保存最佳模型（按 train loss）=====
        if is_main_process and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if getattr(training_args, "use_wandb", True):
                wandb.run.summary["best_train_loss"] = best_loss
            best_dir = os.path.join(training_args.output_dir, "best_model")
            save_model_safe(model.module if hasattr(model, 'module') else model, best_dir)
            print(f"🏆 Best model saved — loss: {best_loss:.6f} — {best_dir}")

    # ===== Barrier: 等待所有进程完成 =====
    if dist_initialized:
        try:
            dist.barrier()
            if is_main_process:
                print("✅ All processes completed, starting cleanup...")
        except Exception as e:
            if is_main_process:
                print(f"⚠️  Warning: dist.barrier() failed: {e}")

    # 最终保存（只在主进程执行）
    if is_main_process:
        final_dir = os.path.join(training_args.output_dir, "final")
        save_model_safe(model.module if hasattr(model, 'module') else model, final_dir)
        if getattr(training_args, "use_wandb", True):
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
            dist.destroy_process_group()
            if is_main_process:
                print("✅ Distributed process group destroyed successfully")
        except Exception as e:
            if is_main_process:
                print(f"⚠️  Warning: Failed to destroy process group: {e}")
    
    if is_main_process:
        print("\n✨ All cleanup complete, exiting...\n")



# ------------------------------
# Main
# ------------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 使用 checkpoint path 作为 model name (离线加载)
    if getattr(model_args, "checkpoint_path", None) is not None:
        model_args.model_name = model_args.checkpoint_path

    train_loop(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
