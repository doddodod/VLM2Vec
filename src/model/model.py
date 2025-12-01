from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.model.processor import QWEN2_5_VL_TOKENSELECTION
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, \
    backbone2model, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V
import os
from src.arguments import ModelArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, INTERNVIDEO2, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, LamRA_QWEN2_5, COLPALI
from src.model.baseline_backbone.colpali import ColPali
from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL
from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_input(self, input):
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            if "input_ids" in input.keys():
                # text side
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)
                return vfeat
        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            # pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            texts = [text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + '\n', '') for text in input["texts"]] # we are actually passing video queries so this should not happen
            images = []
            for imgs in input['images']:
                # if multi images are given, select the middle frame only
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]
                    assert not isinstance(imgs, list) # make sure we have extracted the middle frame and it is no longer a list
                    images.append(imgs)
                else:
                    images.append(imgs)
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)
            return pooled_output
        elif getattr(self, "model_backbone", None) == COLPALI:
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output
        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            input['pixel_values'] = input['pixel_values'].squeeze(dim=1)
            input['image_sizes'] = input['image_sizes'].squeeze(dim=1)
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output
        else:
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from .utils import parse_layer_type
            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        """Load Qwen2-VL style backbone and (optionally) a LoRA adapter.
        
        KEY FIX: During training, the LoRA was applied to base_model wrapped in MMEBModel.
        During inference, we need to replicate that exact structure.
        """
        
        # ===== STEP 1: Determine paths =====
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        backbone_path = model_args.model_name  # 原始 backbone 路径
        lora_path = model_args.checkpoint_path if model_args.checkpoint_path else None  # LoRA 路径（如果有）
        
        # ===== STEP 2: Get backbone type =====
        config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)
        
        print_master(f'Loading backbone [{model_args.model_backbone}] from {backbone_path}')

        # ===== STEP 3: Load base model from ORIGINAL backbone =====
        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
            base_config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
            base_config._attn_implementation = "flash_attention_2"
            base_config.vision_config._attn_implementation = "flash_attention_2"
            
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                backbone_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=base_config
            )
        elif model_args.model_backbone == PHI3V:
            base_config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
            base_config.use_cache = False
            base_config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(
                backbone_path,
                **kwargs, 
                config=base_config,
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            base_model.padding_side = "right"
        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
            base_config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/", trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                "src/model/vlm_backbone/internvideo2/", 
                config=base_config,
                trust_remote_code=True
            )
        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(backbone_path, processor=kwargs['processor'])
            setattr(base_model, 'config', AutoConfig.from_pretrained(backbone_path, trust_remote_code=True))
        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(backbone_path)
            setattr(base_model, 'config', AutoConfig.from_pretrained(backbone_path, trust_remote_code=True))
        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(backbone_path)
            setattr(base_model, 'config', AutoConfig.from_pretrained(backbone_path, trust_remote_code=True))
        elif model_args.model_backbone == COLPALI:
            base_model = ColPali.from_pretrained(backbone_path)
            setattr(base_model, 'config', AutoConfig.from_pretrained(backbone_path, trust_remote_code=True))
        else:
            base_config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=True)
            base_config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                backbone_path,
                **kwargs, 
                config=base_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        # ===== STEP 4: Handle LoRA loading =====
        if model_args.lora and lora_path:
            print_master(f"Loading LoRA from {lora_path}")
            
            if not os.path.isdir(lora_path):
                raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")
            
            # 检查必要文件
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
            adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
            
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")
            if not os.path.exists(adapter_model_path):
                print_master(f"adapter_model.safetensors not found in {lora_path}, trying .bin")
                adapter_model_path = os.path.join(lora_path, "adapter_model.bin")
                if not os.path.exists(adapter_model_path):
                    raise FileNotFoundError(f"adapter_model file not found in {lora_path}")
            
            try:
                # 关键：使用 get_peft_model 来应用 LoRA，和训练时保持一致
                from peft import LoraConfig, get_peft_model
                
                # 从 adapter_config.json 读取 LoRA 配置
                lora_config = LoraConfig.from_pretrained(lora_path)
                print_master(f"LoRA config: r={lora_config.r}, lora_alpha={lora_config.lora_alpha}")
                
                # 应用 LoRA 到 base_model（和训练时一致）
                lora_encoder = get_peft_model(base_model, lora_config)
                
                # 手动加载权重
                from peft.utils import get_peft_model_state_dict
                import safetensors.torch
                
                adapter_weights = safetensors.torch.load_file(
                    os.path.join(lora_path, "adapter_model.safetensors")
                )
                
                # 重命名键以匹配模型结构
                renamed_weights = {}
                skipped_keys = []
                for key, value in adapter_weights.items():
                    # 跳过 magnitude_vector（DoRA 权重），因为模型可能没有这些层
                    if 'lora_magnitude_vector' in key:
                        skipped_keys.append(key)
                        continue
                    
                    # 步骤1：删除多余的嵌套 base_model.model
                    if key.startswith("base_model.model.base_model."):
                        new_key = key.replace("base_model.model.base_model.", "base_model.", 1)
                    else:
                        new_key = key
                    
                    # 步骤2：加上 .default 前缀
                    if '.lora_A.weight' in new_key:
                        new_key = new_key.replace('.lora_A.weight', '.lora_A.default.weight')
                    elif '.lora_B.weight' in new_key:
                        new_key = new_key.replace('.lora_B.weight', '.lora_B.default.weight')
                    
                    renamed_weights[new_key] = value
                
                if skipped_keys:
                    print_master(f"⚠️ Skipped {len(skipped_keys)} DoRA magnitude_vector keys (not available in current model)")
                
                # 加载权重
                lora_encoder.load_state_dict(renamed_weights, strict=False)
                print_master(f"✅ LoRA weights loaded successfully ({len(renamed_weights)} keys loaded)")
                

                
            except Exception as e:
                print_master(f"❌ Failed to load LoRA: {e}")
                import traceback
                print_master(traceback.format_exc())
                lora_encoder = base_model
            
            # 推理模式：合并 LoRA 权重
            if not is_trainable and lora_encoder is not base_model:
                try:
                    lora_encoder = lora_encoder.merge_and_unload()
                    print_master(f"LoRA merged into base model for inference")
                except Exception as e:
                    print_master(f"Warning: merge_and_unload failed: {e}")
            
            model = cls(
                encoder=lora_encoder,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        
        model.model_backbone = model_args.model_backbone
        return model

    # def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
    #     # Loading the base model
    #     model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    #     config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    #     if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
    #         model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
    #         setattr(model_args, 'model_backbone', model_backbone)
    #     print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
    #     if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
    #         config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    #         config._attn_implementation = "flash_attention_2"
    #         config.vision_config._attn_implementation = "flash_attention_2"
    #         base_model = backbone2model[model_args.model_backbone].from_pretrained(
    #             model_args.model_name,
    #             torch_dtype=torch.bfloat16,
    #             low_cpu_mem_usage=True,
    #             config=config
    #         )
    #     elif model_args.model_backbone == PHI3V:
    #         config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    #         config.use_cache = False
    #         config.padding_side = "right"
    #         base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
    #                                                       torch_dtype=torch.bfloat16, trust_remote_code=True)
    #         base_model.padding_side = "right"
    #     elif model_args.model_backbone == INTERNVIDEO2:
    #         print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
    #         config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/",
    #                                             trust_remote_code=True)
    #         base_model = backbone2model[model_args.model_backbone].from_pretrained("src/model/vlm_backbone/internvideo2/", config=config,
    #                                                                                trust_remote_code=True)
    #     elif model_args.model_backbone == GME:
    #         base_model = GmeQwen2VL(model_args.model_name, processor=kwargs['processor'])
    #         setattr(base_model, 'config', config)
    #     elif model_args.model_backbone == LamRA:
    #         base_model = LamRAQwen2VL(model_args.model_name)
    #         setattr(base_model, 'config', config)
    #     elif model_args.model_backbone == LamRA_QWEN2_5:
    #         base_model = LamRAQwen25VL(model_args.model_name)
    #         setattr(base_model, 'config', config)
    #     elif model_args.model_backbone == COLPALI:
    #         base_model = ColPali.from_pretrained(model_args.model_name)
    #         setattr(base_model, 'config', config)
    #     else:
    #         # Loading external base model from HF
    #         config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    #         config.use_cache = False
    #         base_model = cls.TRANSFORMER_CLS.from_pretrained(
    #             model_name_or_path, **kwargs, config=config,
    #             torch_dtype=torch.bfloat16,
    #             trust_remote_code=True)

    #     # Building the model on top of the base
    #     if model_args.lora:
    #         print_master(f'Loading LoRA from {model_name_or_path}')
    #         lora_config = LoraConfig.from_pretrained(model_name_or_path)
    #         lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
    #         lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
    #         if not is_trainable:
    #             lora_model = lora_model.merge_and_unload()
    #         model = cls(
    #             encoder=lora_model,
    #             pooling=model_args.pooling,
    #             normalize=model_args.normalize,
    #             temperature=model_args.temperature
    #         )
    #     else:
    #         model = cls(
    #             encoder=base_model,
    #             pooling=model_args.pooling,
    #             normalize=model_args.normalize,
    #             temperature=model_args.temperature
    #         )

    #     model.model_backbone = model_args.model_backbone
    #     return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    # def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
    #     qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
    #     tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)

    #     if qry_reps is None or tgt_reps is None:
    #         return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

    #     if self.is_ddp:
    #         all_qry_reps = self._dist_gather_tensor(qry_reps)
    #         all_tgt_reps = self._dist_gather_tensor(tgt_reps)
    #     else:
    #         all_qry_reps = qry_reps
    #         all_tgt_reps = tgt_reps

    #     scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
    #     scores = scores.view(all_qry_reps.size(0), -1)
    #     target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    #     target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
    #     loss = self.cross_entropy(scores / self.temperature, target)
    #     if self.is_ddp:
    #         loss = loss * self.world_size

    #     return loss
    def forward(self, qry=None, tgt=None, neg=None, *args, **kwargs):
        """
        qry: query inputs (text+image)
        tgt: positive target inputs (text)
        neg: explicit negative inputs (text)
        """
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None
        neg_reps = self.encode_input(neg) if neg else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps, "neg_reps": neg_reps}

        # DDP gather
        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            all_neg_reps = self._dist_gather_tensor(neg_reps) if neg_reps is not None else None
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps
            all_neg_reps = neg_reps

        batch_size = all_qry_reps.size(0)

        # ===== Positive scores =====
        pos_scores = self.compute_similarity(all_qry_reps, all_tgt_reps)  # [B, 1]

        # ===== Explicit negative scores =====
        if all_neg_reps is not None:
            neg_scores = self.compute_similarity(all_qry_reps, all_neg_reps)  # [B, N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)  # [B, 1+N]
        else:
            scores = pos_scores  # [B, 1]

        # ===== In-batch negative scores =====
        if batch_size > 1:
            batch_sim = self.compute_similarity(all_qry_reps, all_tgt_reps)  # [B, B]
            mask = torch.eye(batch_size, dtype=torch.bool, device=batch_sim.device)
            batch_neg_scores = batch_sim.masked_select(~mask).view(batch_size, -1)  # [B, B-1]
            scores = torch.cat([scores, batch_neg_scores], dim=1)  # [B, 1+N+(B-1)]

        # ===== Cross-entropy loss =====
        target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        loss = self.cross_entropy(scores / self.temperature, target)

        if self.is_ddp:
            loss = loss * self.world_size

        return {"loss": loss, "scores": scores}


    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
