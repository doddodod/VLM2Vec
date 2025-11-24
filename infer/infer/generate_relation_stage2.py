"""
第二阶段：基于第一阶段的高置信度relation结果，生成详细的relation描述（CoT风格）
输入：第一阶段的结果文件（predict_scene_graph_recall.py的输出）
输出：包含生成描述的完整结果

主要功能：
- 为每张图片的每个pair调用一次prompt
- prompt中包含该pair的候选谓词的相对排名（最多10个，按相似度排序）
- 基于相对排名，生成尽可能长的CoT风格的关系描述和总结

支持多GPU多worker和batch优化
"""

import json
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
import warnings
import argparse
import multiprocessing as mp
from multiprocessing import Manager
import math
import time
import traceback

# 设置环境变量抑制transformers警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ========== 配置 ==========
# 第一阶段结果文件（输入）
STAGE1_RESULT_FILE = "/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_20_all.json"

# 原始输入数据文件（用于获取图片路径和物体bbox信息）
INPUT_DATA_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_case_20.json"

# 第二阶段输出文件
STAGE2_OUTPUT_FILE = "/public/home/xiaojw2025/Data/stage2/stage2_generated_results_case_20_qwen2vl.json"

# 生成模型配置
GENERATION_MODEL_PATH = "/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct"

# 生成参数
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_K_RELATIONS = 10  # 已废弃：现在使用top100中的所有候选谓词及其排序名次
BATCH_SIZE = 8  # 批量推理的batch size
SAVE_INTERVAL = 50  # 每处理50个配对保存一次
MEMORY_CLEANUP_INTERVAL = 20  # 每处理20个配对清理一次内存
USE_IMAGE_CACHE = False  # 是否使用图像缓存优化

# ========================

def get_generation_model_class(model_path):
    """根据模型路径自动检测并返回正确的生成模型类"""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = config.model_type if hasattr(config, 'model_type') else None
        
        print(f"🔍 检测到生成模型类型: {model_type}")
        
        if model_type == 'qwen2_vl':
            from transformers import Qwen2VLForConditionalGeneration
            print("✅ 使用 Qwen2VLForConditionalGeneration")
            return Qwen2VLForConditionalGeneration
        elif model_type == 'qwen3_vl':
            try:
                from transformers import Qwen3VLForConditionalGeneration
                print("✅ 使用 Qwen3VLForConditionalGeneration")
                return Qwen3VLForConditionalGeneration
            except ImportError:
                print("⚠️  Qwen3VL 类未找到，尝试使用 AutoModel")
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq
        else:
            print(f"ℹ️  使用通用 AutoModelForVision2Seq (模型类型: {model_type})")
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
            
    except Exception as e:
        print(f"⚠️  模型类型检测失败: {e}")
        print("ℹ️  回退到 AutoModelForVision2Seq")
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq


def configure_attention_backend():
    """配置注意力机制后端"""
    try:
        if not torch.cuda.is_available():
            return "eager"
        
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        compute_capability = major * 10 + minor
        
        if compute_capability >= 80:
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                return "eager"
        else:
            return "eager"
    except Exception as e:
        return "eager"


def normalize_bbox_for_generation(bbox, width, height):
    """将bbox坐标归一化到[0, 1000)范围（用于生成模型）"""
    x1, y1, x2, y2 = bbox
    norm_x1 = int((x1 / width) * 1000)
    norm_y1 = int((y1 / height) * 1000)
    norm_x2 = int((x2 / width) * 1000)
    norm_y2 = int((y2 / height) * 1000)
    
    norm_x1 = max(0, min(norm_x1, 999))
    norm_y1 = max(0, min(norm_y1, 999))
    norm_x2 = max(0, min(norm_x2, 999))
    norm_y2 = max(0, min(norm_y2, 999))
    
    norm_x1, norm_x2 = min(norm_x1, norm_x2), max(norm_x1, norm_x2)
    norm_y1, norm_y2 = min(norm_y1, norm_y2), max(norm_y1, norm_y2)
    
    if norm_x1 == norm_x2:
        norm_x2 = min(norm_x1 + 1, 999)
    if norm_y1 == norm_y2:
        norm_y2 = min(norm_y1 + 1, 999)
    
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def build_prompt(subject_obj, object_obj, ranked_predicates, original_width, original_height):
    """
    构建生成prompt，包含候选谓词的相对排名（最多10个）
    
    Args:
        subject_obj: 主体对象信息
        object_obj: 客体对象信息
        ranked_predicates: list of str，谓词列表（已按相似度排序，最多10个）
        original_width: 图片宽度
        original_height: 图片高度
    """
    # 归一化bbox
    subject_norm_bbox = normalize_bbox_for_generation(
        subject_obj['bbox'], original_width, original_height
    )
    object_norm_bbox = normalize_bbox_for_generation(
        object_obj['bbox'], original_width, original_height
    )
    
    # 格式化bbox字符串（简化格式：x1, y1, x2, y2）
    subject_bbox_str = f"{subject_norm_bbox[0]}, {subject_norm_bbox[1]}, {subject_norm_bbox[2]}, {subject_norm_bbox[3]}"
    object_bbox_str = f"{object_norm_bbox[0]}, {object_norm_bbox[1]}, {object_norm_bbox[2]}, {object_norm_bbox[3]}"
    
    # 构建候选谓词的相对排名列表（1, 2, 3...）
    predicates_text = []
    for idx, predicate in enumerate(ranked_predicates, 1):
        predicates_text.append(f"{idx}. {predicate}")
    
    predicates_text_str = "\n".join(predicates_text)
    
    # 构建prompt（全英文，CoT风格）
    prompt_text = (
        f"In this image, there are two objects:\n"
        f"- <|object_ref_start|>{subject_obj['class_name']}<|object_ref_end|> at <|box_start|>({subject_bbox_str})<|box_end|>\n"
        f"- <|object_ref_start|>{object_obj['class_name']}<|object_ref_end|> at <|box_start|>({object_bbox_str})<|box_end|>\n\n"
        f"Stage 1 predicted candidate predicates for this pair (ranked by similarity, top candidates):\n{predicates_text_str}\n\n"
        f"Based on the ranking information above, please provide a comprehensive and detailed description of the relationship between "
        f"{subject_obj['class_name']} and {object_obj['class_name']}. "
        f"Your description should:\n"
        f"1. Consider the ranking positions of different candidate predicates\n"
        f"2. Consider the visual evidence in the image\n"
        f"3. Provide step-by-step reasoning (chain of thought) about why certain predicates are more likely than others\n"
        f"4. Give a thorough, detailed summary of the relationship that is as comprehensive as possible\n"
        f"5. Ensure your conclusions are well-reasoned and accurate\n\n"
        f"Please write a long, detailed description with clear reasoning steps."
    )
    
    return prompt_text


def parse_generated_text(generated_text):
    """
    从生成的文本中解析CoT风格的描述
    
    Args:
        generated_text: 模型生成的完整文本（CoT风格的描述）
    
    Returns:
        str: 解析后的描述文本
    """
    # 直接返回生成的文本，因为CoT风格的输出应该是一个完整的描述
    # 如果文本为空或太短，返回错误信息
    description = generated_text.strip()
    
    if not description or len(description) < 10:
        return f"Failed to parse description (text too short: {description[:100]}...)"
    
    return description


def generate_relation_for_pair(model, processor, image_path, pair_data, 
                              original_width, original_height):
    """
    为一个配对生成CoT风格的详细描述（使用单个prompt）
    
    Args:
        model: 生成模型
        processor: 生成模型的processor
        image_path: 图片路径
        pair_data: 配对数据，包含:
            - subject_obj: 主体对象信息
            - object_obj: 客体对象信息
            - ranked_predicates: 候选谓词列表（已按相似度排序，最多10个）
        original_width: 图片宽度
        original_height: 图片高度
    
    Returns:
        dict: 包含generated_description的结果
    """
    try:
        # 构建包含排序名次的prompt
        prompt_text = build_prompt(
            pair_data['subject_obj'], 
            pair_data['object_obj'], 
            pair_data['ranked_predicates'],
            original_width, 
            original_height
        )
        
        # 构建conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # 应用chat template
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # 处理输入
        inputs = processor(
            text=text_prompt,
            images=Image.open(image_path),
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 生成（使用较大的max_new_tokens以容纳CoT风格的详细描述）
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS * 2,  # CoT风格需要更多tokens
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        
        generated_text = processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()
        
        # 解析生成的文本
        description = parse_generated_text(generated_text)
        
        # 清理显存
        del inputs, generated_ids
        torch.cuda.empty_cache()
        
        return {
            'generated_description': description
        }
        
    except Exception as e:
        print(f"生成失败: {str(e)}")
        traceback.print_exc()
        # 返回错误结果
        return {
            'generated_description': f"生成失败: {str(e)}"
        }


def generate_relations_batch(model, processor, image_path, generation_tasks, 
                            original_width, original_height, batch_size=BATCH_SIZE, 
                            use_image_cache=USE_IMAGE_CACHE):
    """
    批量生成多个relation描述
    
    Args:
        model: 生成模型
        processor: 生成模型的processor
        image_path: 图片路径
        generation_tasks: list of dict，每个dict包含:
            - subject_obj: 主体对象信息
            - object_obj: 客体对象信息
            - top_predicate: 谓词
            - similarity: 相似度
        original_width: 图片宽度
        original_height: 图片高度
        batch_size: 批量大小
        use_image_cache: 是否使用图像缓存
    
    Returns:
        list of dict: 每个dict包含predicate, similarity, generated_description
    """
    all_results = []
    
    try:
        # 优化：预处理图像一次（如果使用图像缓存）
        cached_pixel_values = None
        cached_image_grid_thw = None
        
        if use_image_cache:
            try:
                dummy_inputs = processor(
                    text=[""],
                    images=[Image.open(image_path)],
                    return_tensors="pt"
                )
                cached_pixel_values = dummy_inputs.get('pixel_values', None)
                cached_image_grid_thw = dummy_inputs.get('image_grid_thw', None)
                
                if cached_pixel_values is not None:
                    cached_pixel_values = cached_pixel_values.to(model.device)
                    if cached_image_grid_thw is not None:
                        cached_image_grid_thw = cached_image_grid_thw.to(model.device)
                
                del dummy_inputs
            except Exception as e:
                use_image_cache = False
        
        # 分批处理
        for i in range(0, len(generation_tasks), batch_size):
            batch = generation_tasks[i:i+batch_size]
            batch_len = len(batch)
            
            # 批量构建prompts
            text_prompts = []
            for task in batch:
                prompt_text = build_prompt(
                    task['subject_obj'], task['object_obj'], task['top_predicate'],
                    original_width, original_height
                )
                
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                
                text_prompt = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                text_prompts.append(text_prompt)
            
            # 批量处理输入
            if use_image_cache and cached_pixel_values is not None:
                text_inputs = processor(
                    text=text_prompts,
                    images=None,
                    return_tensors="pt",
                    padding=True
                )
                text_inputs['pixel_values'] = cached_pixel_values.repeat(batch_len, 1, 1, 1)
                if cached_image_grid_thw is not None:
                    text_inputs['image_grid_thw'] = cached_image_grid_thw.repeat(batch_len, 1)
                
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) and v.device != model.device else v 
                         for k, v in text_inputs.items()}
            else:
                inputs = processor(
                    text=text_prompts,
                    images=[Image.open(image_path)] * batch_len,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # 批量生成
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # 批量解码
            for j, gen_id in enumerate(generated_ids):
                input_length = inputs["input_ids"][j].shape[0]
                generated_tokens = gen_id[input_length:]
                
                generated_text = processor.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()
                
                all_results.append({
                    'predicate': batch[j]['top_predicate'],
                    'similarity': batch[j]['similarity'],
                    'generated_description': generated_text
                })
            
            # 释放显存
            del inputs, generated_ids
            torch.cuda.empty_cache()
        
        return all_results
        
    except Exception as e:
        print(f"批量生成出错: {str(e)}")
        traceback.print_exc()
        # 返回错误结果
        return [{
            'predicate': task['top_predicate'],
            'similarity': task['similarity'],
            'generated_description': f"生成失败: {str(e)}"
        } for task in generation_tasks]


def load_existing_results(output_path):
    """加载已存在的结果"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None


def get_processed_pairs(existing_results):
    """从已存在结果中提取已处理的配对（使用物体ID区分同名物体）"""
    processed_pairs = set()
    if existing_results and 'results' in existing_results:
        for result in existing_results['results']:
            # 优先使用物体ID，如果没有则使用类别名（向后兼容）
            subject_id = result.get('subject_id', None)
            object_id = result.get('object_id', None)
            if subject_id is not None and object_id is not None:
                pair_key = (result['image_id'], subject_id, object_id)
            else:
                pair_key = (result['image_id'], result['subject'], result['object'])
            processed_pairs.add(pair_key)
    return processed_pairs


def split_data(data, num_splits):
    """将数据均衡分割成num_splits份"""
    if num_splits <= 1:
        return [data]
    
    total = len(data)
    if total == 0:
        # 如果数据为空，返回num_splits个空列表
        return [[] for _ in range(num_splits)]
    
    chunk_size = math.ceil(total / num_splits)
    chunks = []
    
    for i in range(num_splits):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        if start < total:
            chunks.append(data[start:end])
        else:
            # 确保总是返回num_splits个chunks，即使有些是空的
            chunks.append([])
    
    return chunks


def inference_on_gpu(gpu_id, data_chunk, model_path, output_prefix,
                     shared_stats, batch_size=BATCH_SIZE,
                     worker_id=None, max_memory=None):
    """在指定GPU上执行推理"""
    start_time = time.time()
    
    if worker_id is not None:
        print(f"\n[GPU {gpu_id} Worker {worker_id}] 开始加载模型...")
    else:
        print(f"\n[GPU {gpu_id}] 开始加载模型...")
    
    torch.cuda.set_device(gpu_id)
    
    try:
        GenModelClass = get_generation_model_class(model_path)
        attn_implementation = configure_attention_backend()
        
        from transformers import AutoProcessor, AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, '_attn_implementation'):
            config._attn_implementation = attn_implementation
        
        # 加载模型
        if max_memory is not None:
            max_memory_dict = {gpu_id: f"{max_memory}MB"}
            model = GenModelClass.from_pretrained(
                model_path,
                device_map=f"cuda:{gpu_id}",
                max_memory=max_memory_dict,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                config=config
            )
        else:
            model = GenModelClass.from_pretrained(
                model_path,
                device_map=f"cuda:{gpu_id}",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                config=config
            )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model.eval()
        
        if worker_id is not None:
            print(f"[GPU {gpu_id} Worker {worker_id}] ✓ 模型加载完成，处理 {len(data_chunk)} 个配对")
        else:
            print(f"[GPU {gpu_id}] ✓ 模型加载完成，处理 {len(data_chunk)} 个配对")
        
        # 检查断点续传
        if worker_id is not None:
            gpu_output_path = f"{output_prefix}_gpu{gpu_id}_worker{worker_id}.json"
        else:
            gpu_output_path = f"{output_prefix}_gpu{gpu_id}.json"
        
        existing_results = load_existing_results(gpu_output_path)
        processed_pairs = get_processed_pairs(existing_results)
        
        if processed_pairs:
            if worker_id is not None:
                print(f"[GPU {gpu_id} Worker {worker_id}] ✓ 发现已处理结果: {len(processed_pairs)} 个配对")
            else:
                print(f"[GPU {gpu_id}] ✓ 发现已处理结果: {len(processed_pairs)} 个配对")
        
        # 过滤未处理的配对（使用物体ID区分同名物体）
        unprocessed_chunk = []
        for pair_data in data_chunk:
            # 优先使用物体ID，如果没有则使用类别名（向后兼容）
            subject_id = pair_data.get('subject_id', None)
            object_id = pair_data.get('object_id', None)
            if subject_id is not None and object_id is not None:
                pair_key = (pair_data['image_id'], subject_id, object_id)
            else:
                pair_key = (pair_data['image_id'], pair_data['subject'], pair_data['object'])
            if pair_key not in processed_pairs:
                unprocessed_chunk.append(pair_data)
        
        if not unprocessed_chunk:
            if worker_id is not None:
                print(f"[GPU {gpu_id} Worker {worker_id}] ✓ 所有配对已处理完成")
            else:
                print(f"[GPU {gpu_id}] ✓ 所有配对已处理完成")
            return
        
        # 加载已存在的结果
        all_results = existing_results.get('results', []) if existing_results else []
        processed_count = len(processed_pairs)
        error_count = 0
        
        # 处理未处理的配对
        for pair_data in tqdm(unprocessed_chunk, desc=f"GPU{gpu_id}" + (f"W{worker_id}" if worker_id is not None else "")):
            try:
                image_path = pair_data['image_path']
                if not os.path.exists(image_path):
                    continue
                
                with Image.open(image_path) as img:
                    original_width, original_height = img.size
                
                # 生成CoT风格的详细描述
                stage2_results = generate_relation_for_pair(
                    model, processor, image_path, pair_data,
                    original_width, original_height
                )
                
                # 保存结果（包含物体ID以区分同名物体）
                result = {
                    'image_id': pair_data['image_id'],
                    'subject_id': pair_data.get('subject_id', None),  # 添加subject_id
                    'object_id': pair_data.get('object_id', None),  # 添加object_id
                    'subject': pair_data['subject'],
                    'object': pair_data['object'],
                    'ranked_predicates': pair_data['ranked_predicates'],
                    'stage2_generated_description': stage2_results['generated_description'],
                    'has_gt': pair_data.get('has_gt', False),
                    'gt_predicates': pair_data.get('gt_predicates', [])
                }
                
                all_results.append(result)
                processed_count += 1
                
                # 定期保存
                if processed_count % SAVE_INTERVAL == 0:
                    output_data = {
                        'summary': {
                            'total_pairs': len(all_results),
                            'processed_pairs': processed_count
                        },
                        'results': all_results
                    }
                    temp_output_path = f"{output_prefix}_gpu{gpu_id}_worker{worker_id}_temp.json" if worker_id is not None else f"{output_prefix}_gpu{gpu_id}_temp.json"
                    with open(temp_output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                # 定期清理显存
                if processed_count % MEMORY_CLEANUP_INTERVAL == 0:
                    torch.cuda.empty_cache()
            
            except Exception as e:
                error_count += 1
                if worker_id is not None:
                    print(f"\n[GPU {gpu_id} Worker {worker_id}] 处理配对 {pair_data.get('image_id', 'unknown')} 出错: {str(e)}")
                else:
                    print(f"\n[GPU {gpu_id}] 处理配对 {pair_data.get('image_id', 'unknown')} 出错: {str(e)}")
        
        # 保存最终结果
        output_data = {
            'summary': {
                'total_pairs': len(all_results),
                'processed_pairs': processed_count,
                'error_count': error_count
            },
            'results': all_results
        }
        
        with open(gpu_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # 更新统计信息
        stats_key = f"{gpu_id}_worker{worker_id}" if worker_id is not None else str(gpu_id)
        shared_stats[stats_key] = {
            'processed': processed_count,
            'errors': error_count,
            'time_minutes': (time.time() - start_time) / 60
        }
        
        if worker_id is not None:
            print(f"\n[GPU {gpu_id} Worker {worker_id}] ✓ 推理完成！处理了 {processed_count} 个配对，失败 {error_count} 个")
        else:
            print(f"\n[GPU {gpu_id}] ✓ 推理完成！处理了 {processed_count} 个配对，失败 {error_count} 个")
    
    except Exception as e:
        print(f"\n[GPU {gpu_id}] 严重错误: {str(e)}")
        traceback.print_exc()
        stats_key = f"{gpu_id}_worker{worker_id}" if worker_id is not None else str(gpu_id)
        shared_stats[stats_key] = {
            'processed': 0,
            'errors': len(data_chunk),
            'error_msg': str(e)
        }


def merge_results(output_prefix, num_gpus, final_output_path, total_pairs, workers_per_gpu=1):
    """合并所有GPU的结果"""
    print(f"\n合并 {num_gpus} 个GPU的结果（每GPU {workers_per_gpu} 个worker）...")
    
    all_results = []
    processed_pairs = set()
    
    for gpu_id in range(num_gpus):
        if workers_per_gpu > 1:
            for worker_id in range(workers_per_gpu):
                gpu_output_path = f"{output_prefix}_gpu{gpu_id}_worker{worker_id}.json"
                if os.path.exists(gpu_output_path):
                    with open(gpu_output_path, 'r', encoding='utf-8') as f:
                        gpu_results = json.load(f)
                        for result in gpu_results.get('results', []):
                            # 优先使用物体ID，如果没有则使用类别名（向后兼容）
                            subject_id = result.get('subject_id', None)
                            object_id = result.get('object_id', None)
                            if subject_id is not None and object_id is not None:
                                pair_key = (result['image_id'], subject_id, object_id)
                            else:
                                pair_key = (result['image_id'], result['subject'], result['object'])
                            if pair_key not in processed_pairs:
                                all_results.append(result)
                                processed_pairs.add(pair_key)
        else:
            gpu_output_path = f"{output_prefix}_gpu{gpu_id}.json"
            if os.path.exists(gpu_output_path):
                with open(gpu_output_path, 'r', encoding='utf-8') as f:
                    gpu_results = json.load(f)
                    for result in gpu_results.get('results', []):
                        # 优先使用物体ID，如果没有则使用类别名（向后兼容）
                        subject_id = result.get('subject_id', None)
                        object_id = result.get('object_id', None)
                        if subject_id is not None and object_id is not None:
                            pair_key = (result['image_id'], subject_id, object_id)
                        else:
                            pair_key = (result['image_id'], result['subject'], result['object'])
                        if pair_key not in processed_pairs:
                            all_results.append(result)
                            processed_pairs.add(pair_key)
    
    # 保存合并结果
    output_data = {
        'summary': {
            'total_pairs': len(all_results),
            'total_images': len(set(r['image_id'] for r in all_results)),
            'top_k_relations': TOP_K_RELATIONS,
            'generation_max_tokens': MAX_NEW_TOKENS,
            'generation_temperature': TEMPERATURE,
            'num_gpus': num_gpus,
            'workers_per_gpu': workers_per_gpu
        },
        'results': all_results
    }
    
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 合并完成，共 {len(all_results)} 个配对")


def prepare_data_for_inference(stage1_data, input_data_map):
    """准备推理数据"""
    per_image_top100 = stage1_data.get('per_image_top100_candidates', {})
    
    all_pairs = []
    
    for image_id, top100_candidates in per_image_top100.items():
        # 尝试多种格式匹配image_id
        img_data = None
        if image_id in input_data_map:
            img_data = input_data_map[image_id]
        elif str(image_id) in input_data_map:
            img_data = input_data_map[str(image_id)]
        elif isinstance(image_id, str) and image_id.isdigit() and int(image_id) in input_data_map:
            img_data = input_data_map[int(image_id)]
        
        if img_data is None:
            continue
        
        image_path = img_data['image_path']
        objects = img_data['objects']
        
        if not os.path.exists(image_path):
            continue
        
        # 创建物体名称到物体信息的映射（用于向后兼容）
        obj_dict_by_name = {obj['class_name']: obj for obj in objects}
        # 创建物体ID到物体信息的映射
        obj_dict_by_id = {obj['id']: obj for obj in objects}
        
        # 按配对分组候选（使用物体ID区分同名物体）
        pair_candidates = {}
        for candidate in top100_candidates:
            # 优先使用物体ID
            subject_id = candidate.get('subject_id', None)
            object_id = candidate.get('object_id', None)
            subject = candidate['subject']
            object_name = candidate['object']
            predicate = candidate.get('predicted_predicate', 'no relation')
            
            # 使用物体ID作为key（如果可用），否则使用类别名（向后兼容）
            if subject_id is not None and object_id is not None:
                pair_key = (subject_id, object_id)
            else:
                pair_key = (subject, object_name)
            
            if pair_key not in pair_candidates:
                pair_candidates[pair_key] = {
                    'candidates': [],
                    'subject_id': subject_id,
                    'object_id': object_id,
                    'subject': subject,
                    'object': object_name
                }
            
            if predicate != 'no relation':
                pair_candidates[pair_key]['candidates'].append({
                    'predicate': predicate,
                    'similarity': candidate.get('similarity', 0)
                })
        
        # 对每个配对准备数据
        for pair_key, pair_data in pair_candidates.items():
            candidates = pair_data['candidates']
            subject_id = pair_data['subject_id']
            object_id = pair_data['object_id']
            subject_name = pair_data['subject']
            object_name = pair_data['object']
            
            if not candidates:
                continue
            
            # 获取物体对象信息
            if subject_id is not None and object_id is not None:
                # 使用物体ID获取
                if subject_id not in obj_dict_by_id or object_id not in obj_dict_by_id:
                    continue
                subject_obj = obj_dict_by_id[subject_id]
                object_obj = obj_dict_by_id[object_id]
            else:
                # 向后兼容：使用类别名获取
                if subject_name not in obj_dict_by_name or object_name not in obj_dict_by_name:
                    continue
                subject_obj = obj_dict_by_name[subject_name]
                object_obj = obj_dict_by_name[object_name]
                # 从对象中获取ID
                subject_id = subject_obj.get('id', None)
                object_id = object_obj.get('id', None)
            
            # 按相似度排序，取前10个（最高的10个）
            candidates_sorted = sorted(
                candidates,
                key=lambda x: x['similarity'],
                reverse=True
            )
            top_predicates = candidates_sorted[:10]  # 最多10个
            
            # 只保存predicate，不需要rank和similarity
            ranked_predicates = [item['predicate'] for item in top_predicates]
            
            # 获取GT信息（从第一个候选获取）
            has_gt = False
            gt_predicates = []
            for candidate in top100_candidates:
                # 匹配条件：优先使用物体ID，否则使用类别名
                match = False
                if subject_id is not None and object_id is not None:
                    if (candidate.get('subject_id') == subject_id and 
                        candidate.get('object_id') == object_id):
                        match = True
                else:
                    if (candidate['subject'] == subject_name and 
                        candidate['object'] == object_name):
                        match = True
                
                if match:
                    has_gt = candidate.get('has_gt', False)
                    gt_predicates = candidate.get('gt_predicates', [])
                    break
            
            all_pairs.append({
                'image_id': image_id,
                'image_path': image_path,
                'subject_id': subject_id,  # 添加subject_id
                'object_id': object_id,  # 添加object_id
                'subject': subject_name,
                'object': object_name,
                'subject_obj': subject_obj,
                'object_obj': object_obj,
                'ranked_predicates': ranked_predicates,  # 只包含predicate列表（最多10个，按相似度排序）
                'has_gt': has_gt,
                'gt_predicates': gt_predicates
            })
    
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description='第二阶段：生成详细relation描述')
    parser.add_argument('--stage1_result', type=str, default=STAGE1_RESULT_FILE,
                       help='第一阶段结果文件路径')
    parser.add_argument('--input_data', type=str, default=INPUT_DATA_FILE,
                       help='原始输入数据文件路径')
    parser.add_argument('--output', type=str, default=STAGE2_OUTPUT_FILE,
                       help='输出文件路径')
    parser.add_argument('--model_path', type=str, default=GENERATION_MODEL_PATH,
                       help='生成模型路径')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='使用的GPU数量，默认为1（单GPU模式）')
    
    # 先声明global，再使用这些变量
    global TOP_K_RELATIONS, BATCH_SIZE
    
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'批量推理的batch size，默认{BATCH_SIZE}')
    parser.add_argument('--workers_per_gpu', type=int, default=1,
                       help='每个GPU上的工作进程数，默认1')
    parser.add_argument('--top_k', type=int, default=TOP_K_RELATIONS,
                       help=f'每个配对选择的高置信度relation数量，默认{TOP_K_RELATIONS}')
    
    args = parser.parse_args()
    
    # 更新全局变量
    TOP_K_RELATIONS = args.top_k
    BATCH_SIZE = args.batch_size
    
    print("="*80)
    print("第二阶段：基于第一阶段结果生成详细relation描述")
    print("="*80)
    print(f"✓ Stage1结果文件: {args.stage1_result}")
    print(f"✓ 输入数据文件: {args.input_data}")
    print(f"✓ 输出文件: {args.output}")
    print(f"✓ 模型路径: {args.model_path}")
    print(f"✓ GPU数量: {args.num_gpus}")
    print(f"✓ Batch Size: {args.batch_size}")
    print(f"✓ Workers per GPU: {args.workers_per_gpu}")
    print(f"✓ Top-K Relations: {args.top_k}")
    
    # 加载第一阶段结果
    print(f"\n📖 正在加载第一阶段结果...")
    with open(args.stage1_result, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    
    # 加载原始输入数据
    print(f"📖 正在加载原始输入数据...")
    with open(args.input_data, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 创建image_id映射，支持字符串和整数类型的匹配
    image_data_map = {}
    for img in input_data:
        img_id = img['image_id']
        # 同时支持字符串和整数类型的key
        image_data_map[str(img_id)] = img
        if isinstance(img_id, int):
            image_data_map[img_id] = img
        elif isinstance(img_id, str) and img_id.isdigit():
            image_data_map[int(img_id)] = img
    
    # 准备推理数据
    print(f"📖 正在准备推理数据...")
    all_pairs = prepare_data_for_inference(stage1_data, image_data_map)
    print(f"   共 {len(all_pairs)} 个配对需要处理")
    
    # 根据GPU数量选择推理模式
    if args.num_gpus == 1:
        # 单GPU模式
        print(f"\n[3/3] 单GPU推理模式")
        print("-" * 80)
        
        GenModelClass = get_generation_model_class(args.model_path)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = GenModelClass.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # 检查断点续传
        existing_results = load_existing_results(args.output)
        processed_pairs = get_processed_pairs(existing_results)
        
        if processed_pairs:
            print(f"✓ 发现已处理结果: {len(processed_pairs)} 个配对")
        
        unprocessed_pairs = []
        for p in all_pairs:
            # 优先使用物体ID，如果没有则使用类别名（向后兼容）
            subject_id = p.get('subject_id', None)
            object_id = p.get('object_id', None)
            if subject_id is not None and object_id is not None:
                pair_key = (p['image_id'], subject_id, object_id)
            else:
                pair_key = (p['image_id'], p['subject'], p['object'])
            if pair_key not in processed_pairs:
                unprocessed_pairs.append(p)
        
        if not unprocessed_pairs:
            print("✓ 所有配对已处理完成")
            return
        
        all_results = existing_results.get('results', []) if existing_results else []
        processed_count = len(processed_pairs)
        
        for pair_data in tqdm(unprocessed_pairs, desc="处理配对"):
            try:
                image_path = pair_data['image_path']
                with Image.open(image_path) as img:
                    original_width, original_height = img.size
                
                # 生成CoT风格的详细描述
                stage2_results = generate_relation_for_pair(
                    model, processor, image_path, pair_data,
                    original_width, original_height
                )
                
                result = {
                    'image_id': pair_data['image_id'],
                    'subject_id': pair_data.get('subject_id', None),  # 添加subject_id
                    'object_id': pair_data.get('object_id', None),  # 添加object_id
                    'subject': pair_data['subject'],
                    'object': pair_data['object'],
                    'ranked_predicates': pair_data['ranked_predicates'],
                    'stage2_generated_description': stage2_results['generated_description'],
                    'has_gt': pair_data.get('has_gt', False),
                    'gt_predicates': pair_data.get('gt_predicates', [])
                }
                
                all_results.append(result)
                processed_count += 1
                
                if processed_count % SAVE_INTERVAL == 0:
                    output_data = {
                        'summary': {
                            'total_pairs': len(all_results),
                            'processed_pairs': processed_count
                        },
                        'results': all_results
                    }
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            except Exception as e:
                print(f"处理配对出错: {str(e)}")
        
        # 保存最终结果
        output_data = {
            'summary': {
                'total_pairs': len(all_results),
                'top_k_relations': TOP_K_RELATIONS,
                'generation_max_tokens': MAX_NEW_TOKENS,
                'generation_temperature': TEMPERATURE
            },
            'results': all_results
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 推理完成！处理了 {processed_count} 个配对")
    
    else:
        # 多GPU模式
        total_workers = args.num_gpus * args.workers_per_gpu
        print(f"\n[3/3] 多GPU推理模式 ({args.num_gpus} 个GPU, 每GPU {args.workers_per_gpu} 个worker, 共 {total_workers} 个进程)")
        print("-" * 80)
        
        output_prefix = args.output.replace('.json', '')
        
        # 检查断点续传
        final_existing_results = load_existing_results(args.output)
        if final_existing_results:
            final_processed_pairs = get_processed_pairs(final_existing_results)
            print(f"✓ 发现最终合并结果: {len(final_processed_pairs)} 个配对")
            filtered_pairs = []
            for p in all_pairs:
                # 优先使用物体ID，如果没有则使用类别名（向后兼容）
                subject_id = p.get('subject_id', None)
                object_id = p.get('object_id', None)
                if subject_id is not None and object_id is not None:
                    pair_key = (p['image_id'], subject_id, object_id)
                else:
                    pair_key = (p['image_id'], p['subject'], p['object'])
                if pair_key not in final_processed_pairs:
                    filtered_pairs.append(p)
            all_pairs = filtered_pairs
            print(f"✓ 过滤后剩余未处理配对: {len(all_pairs)} 个")
            if len(all_pairs) == 0:
                print("✓ 所有配对已处理完成")
                return
        
        # 分割数据
        data_chunks = split_data(all_pairs, args.num_gpus)
        
        # 确保data_chunks的长度等于num_gpus
        if len(data_chunks) < args.num_gpus:
            # 如果chunks数量不足，补充空列表
            while len(data_chunks) < args.num_gpus:
                data_chunks.append([])
        
        # 如果每GPU有多个worker，需要进一步分割数据
        if args.workers_per_gpu > 1:
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                max_memory_per_worker = int(gpu_memory_mb * 0.8 / args.workers_per_gpu)
                print(f"✓ 每GPU显存: {gpu_memory_mb/1024:.1f}GB, 每Worker显存限制: {max_memory_per_worker/1024:.1f}GB")
            else:
                max_memory_per_worker = None
            
            worker_chunks = []
            worker_gpu_ids = []
            worker_ids = []
            for gpu_id in range(args.num_gpus):
                if len(data_chunks[gpu_id]) > 0:
                    gpu_data = data_chunks[gpu_id]
                    worker_data_chunks = split_data(gpu_data, args.workers_per_gpu)
                    for worker_id in range(args.workers_per_gpu):
                        if len(worker_data_chunks[worker_id]) > 0:
                            worker_chunks.append(worker_data_chunks[worker_id])
                            worker_gpu_ids.append(gpu_id)
                            worker_ids.append(worker_id)
        else:
            worker_chunks = []
            worker_gpu_ids = []
            worker_ids = []
            max_memory_per_worker = None
            for gpu_id in range(args.num_gpus):
                if len(data_chunks[gpu_id]) > 0:
                    worker_chunks.append(data_chunks[gpu_id])
                    worker_gpu_ids.append(gpu_id)
                    worker_ids.append(None)
        
        print(f"✓ 数据已分割成 {len(worker_chunks)} 份")
        
        # 使用multiprocessing启动多个进程
        manager = Manager()
        shared_stats = manager.dict()
        
        processes = []
        start_time = time.time()
        
        for gpu_id, worker_id, chunk in zip(worker_gpu_ids, worker_ids, worker_chunks):
            p = mp.Process(
                target=inference_on_gpu,
                args=(gpu_id, chunk, args.model_path, output_prefix,
                     shared_stats, args.batch_size, worker_id, max_memory_per_worker)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print("\n" + "=" * 80)
        print("所有GPU推理完成，开始合并结果...")
        print("=" * 80)
        
        # 合并结果
        merge_results(output_prefix, args.num_gpus, args.output, len(all_pairs), args.workers_per_gpu)
        
        # 输出最终统计
        total_time = time.time() - start_time
        total_processed = sum(stats.get("processed", 0) for stats in shared_stats.values())
        total_errors = sum(stats.get("errors", 0) for stats in shared_stats.values())
        
        print("\n" + "=" * 80)
        print("多GPU推理完成！")
        print("=" * 80)
        print(f"总配对数: {len(all_pairs)}")
        print(f"成功处理: {total_processed}")
        print(f"失败数: {total_errors}")
        print(f"总耗时: {total_time/60:.2f}分钟")
        print(f"总进程数: {total_workers}")
        print("=" * 80)
        print(f"✓ 最终结果保存至: {args.output}")


if __name__ == "__main__":
    # 在Linux上，必须使用spawn方法才能在多进程中正确使用CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
