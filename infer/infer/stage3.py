import json
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
import warnings
from multiprocessing import Process, Queue, Manager
import math
import argparse

# 添加项目根目录到Python路径，以便导入src模块
# 脚本位于 embedding/infer/ 目录下，需要向上两级找到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def check_flash_attention_support():

    try:
        # 检查是否有可用的GPU
        if not torch.cuda.is_available():
            return False, "CUDA不可用"
        
        # 获取GPU计算能力
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        compute_capability = major * 10 + minor
        
        # Flash Attention 2需要计算能力 >= 8.0 (Ampere及以上架构)
        # Flash Attention 1需要计算能力 >= 7.5 (Turing及以上架构)
        if compute_capability >= 80:
            # 尝试导入flash_attn
            try:
                import flash_attn
                return True, f"支持Flash Attention (GPU计算能力: {major}.{minor})"
            except ImportError:
                return False, f"GPU支持但未安装flash_attn包 (计算能力: {major}.{minor})"
        else:
            return False, f"GPU计算能力不足 (当前: {major}.{minor}, 需要: >= 8.0)"
            
    except Exception as e:
        return False, f"检测失败: {str(e)}"


def configure_attention_backend():

    is_supported, message = check_flash_attention_support()
    
    print("\n" + "="*80)
    print("注意力机制配置")
    print("="*80)
    
    if is_supported:
        print(f"✅ {message}")
        print("   使用: Flash Attention (最快)")
        os.environ["ATTN_IMPLEMENTATION"] = "flash_attention_2"
        # 同时设置transformers使用的环境变量
        os.environ["USE_FLASH_ATTENTION"] = "1"
        return "flash_attn"
    else:
        print(f"⚠️  {message}")
        
        # 检查PyTorch版本是否支持SDPA
        pytorch_version = torch.__version__
        major, minor = map(int, pytorch_version.split('.')[:2])
        
        if major >= 2:  # PyTorch 2.0+支持SDPA
            print("   降级使用: Scaled Dot Product Attention (SDPA)")
            print("   性能: 中等，但比eager模式快")
            os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
            os.environ["USE_FLASH_ATTENTION"] = "0"
            return "sdpa"
        else:
            print("   降级使用: Eager Attention (标准实现)")
            print("   性能: 较慢，但兼容性最好")
            os.environ["ATTN_IMPLEMENTATION"] = "eager"
            os.environ["USE_FLASH_ATTENTION"] = "0"
            return "eager"
    
    print("="*80 + "\n")


_attn_type = configure_attention_backend()

# 现在才导入VLM2Vec模块
from src.model.model import MMEBModel
from src.arguments import ModelArguments, DataArguments
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS


# Stage2输出文件（包含CoT数据）
STAGE2_OUTPUT_FILE = "/public/home/xiaojw2025/Data/stage2/stage2_generated_results.json"
INPUT_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_2000_images.json"
OUTPUT_FILE = "/public/home/xiaojw2025/Data/stage3/recall_results_2000_stage3.json"

# 默认使用的GPU数量（None表示使用所有可用GPU）
# 也可以通过命令行参数 --num_gpus 或环境变量 NUM_GPUS 指定
NUM_GPUS = None  # 设置为 None 使用所有GPU，或设置为具体数字如 2 表示只使用2个GPU

# 50个谓词列表
PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind",
    "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
    "in", "in front of", "laying on", "looking at", "lying on", "made of",
    "mounted on", "near", "of", "on", "on back of", "over", "painted on",
    "parked on", "part of", "playing", "riding", "says", "sitting on",
    "standing on", "to", "under", "using", "walking in", "walking on",
    "watching", "wearing", "wears", "with"
]


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
    return f"<|object_ref_start|>{object_label}<|object_ref_end|>"


def load_stage2_cot_data(stage2_file):
    """
    加载stage2的输出文件，建立(image_id, subject_id, object_id) -> CoT描述的映射
    同时支持使用类别名的向后兼容格式
    
    Returns:
        dict: {(image_id, subject_id, object_id): cot_description} 或 {(image_id, subject, object): cot_description}
    """
    print(f"📖 正在加载Stage2 CoT数据: {stage2_file}")
    with open(stage2_file, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)
    
    cot_map = {}
    results = stage2_data.get('results', [])
    
    for result in results:
        image_id = result['image_id']
        subject_id = result.get('subject_id', None)
        object_id = result.get('object_id', None)
        subject = result['subject'].strip()  # 去除可能的空格
        object_name = result['object'].strip()  # 去除可能的空格
        cot_description = result.get('stage2_generated_description', '')
        
        # 优先使用物体ID创建key（区分同名物体）
        if subject_id is not None and object_id is not None:
            # 使用物体ID作为key
            key_formats = [
                (str(image_id), subject_id, object_id),  # 字符串image_id
                (image_id, subject_id, object_id),  # 原始image_id格式
            ]
            # 如果image_id是字符串且可转换为整数，也添加整数key
            if isinstance(image_id, str) and image_id.isdigit():
                key_formats.append((int(image_id), subject_id, object_id))
            # 如果image_id是整数，也添加字符串key（上面已添加）
            
            for key_format in key_formats:
                cot_map[key_format] = cot_description
        else:
            # 向后兼容：使用类别名作为key
            key_formats = [
                (str(image_id), subject, object_name),  # 字符串image_id
                (image_id, subject, object_name),  # 原始image_id格式
            ]
            if isinstance(image_id, str) and image_id.isdigit():
                key_formats.append((int(image_id), subject, object_name))
            
            for key_format in key_formats:
                cot_map[key_format] = cot_description
    
    unique_entries = len(results)
    total_keys = len(cot_map)
    print(f"   ✓ 加载了 {unique_entries} 个CoT描述（共 {total_keys} 个key格式）")
    return cot_map


def precompute_predicate_vectors(model, processor, predicates, device='cuda', progress_queue=None, gpu_id=None):
    """
    预计算所有谓词的向量表示（只需要计算一次）
    
    Args:
        model: VLM2Vec模型
        processor: 文本处理器
        predicates: 谓词列表
        device: 设备名称，如 'cuda:0'
        progress_queue: 进度队列（可选，用于多进程环境）
        gpu_id: GPU ID（可选，用于多进程环境）
    
    Returns:
        predicate_vectors: [num_predicates, hidden_dim] 的tensor
    """
    if progress_queue is not None and gpu_id is not None:
        progress_queue.put((gpu_id, f"GPU{gpu_id}: 开始预计算谓词向量..."))
    
    predicate_vectors = []
    
    # 在多进程环境下，不使用tqdm（输出会被缓冲），改用简单打印
    for idx, predicate in enumerate(predicates):
        predicate_text = f"The subject is {predicate} the object."
        inputs = processor(text=predicate_text, images=None, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            tgt_output = model(tgt=inputs)["tgt_reps"]
            predicate_vectors.append(tgt_output)
        
        # 每10个谓词更新一次进度
        if (idx + 1) % 10 == 0 and progress_queue is not None and gpu_id is not None:
            progress_queue.put((gpu_id, f"GPU{gpu_id}: 预计算谓词向量进度: {idx + 1}/{len(predicates)}"))
    
    # 堆叠成一个tensor: [num_predicates, hidden_dim]
    predicate_vectors = torch.cat(predicate_vectors, dim=0)
    
    if progress_queue is not None and gpu_id is not None:
        progress_queue.put((gpu_id, f"GPU{gpu_id}: 谓词向量预计算完成，shape: {predicate_vectors.shape}"))
    
    return predicate_vectors


# 全局标志，用于跟踪是否是第一次推理
_first_inference_printed = False

def predict_relation(model, processor, image_path, subject_obj, object_obj, 
                     original_width, original_height, cot_description, predicate_vectors=None, device='cuda',
                     use_original_query=False, use_image=False):
    """
    预测关系，使用stage2的CoT描述代替原始query
    
    Args:
        model: VLM2Vec模型
        processor: 文本处理器
        image_path: 图片路径
        subject_obj: 主体对象信息
        object_obj: 客体对象信息
        original_width: 图片宽度
        original_height: 图片高度
        cot_description: stage2生成的CoT描述文本
        predicate_vectors: 预计算的谓词向量 [num_predicates, hidden_dim]，如果为None则实时计算
        device: 设备名称，如 'cuda:0'
        use_original_query: 是否在cot_description前加上原始query
        use_image: 是否调用图像
    """
    global _first_inference_printed
    
    # 只使用CoT描述，如果为空则返回None（跳过该样本）
    if not cot_description or not cot_description.strip():
        return None
    
    # 构建查询文本
    query_text = cot_description.strip()
    
    # 如果需要添加原始query
    if use_original_query:
        # 构建原始query
        subj_ref = format_object_with_ref(subject_obj['class_name'])
        obj_ref = format_object_with_ref(object_obj['class_name'])
        subj_bbox_token = format_bbox_as_special_token(
            subject_obj['bbox'], 
            normalize=True, 
            original_width=original_width, 
            original_height=original_height
        )
        obj_bbox_token = format_bbox_as_special_token(
            object_obj['bbox'], 
            normalize=True, 
            original_width=original_width, 
            original_height=original_height
        )
        # 构建坐标信息前缀（与predict_scene_graph_recall_stage3.py保持一致）
        coordinate_prefix = f"In the given image, the subject {subj_ref} is located at {subj_bbox_token},the object{obj_ref} is located at {obj_bbox_token}. Please describe the predicate relationship between the subject and the object as the subject is *predicate* the object.Besides,"
        
        # 如果使用图像，添加图像token（图像token在最前面）
        image_token = ""
        if use_image:
            image_token = VLM_IMAGE_TOKENS[QWEN2_VL]
            if image_token:
                image_token = f"{image_token} "
        
        # 拼接：图像token + 坐标信息 + CoT描述（直接拼接，不使用换行符）
        query_text = f"{image_token}{coordinate_prefix}{query_text}"
    
    # 决定是否加载图像
    image = None
    if use_image:
        # 如果使用图像，但文本中没有图像token（且use_original_query=False），需要添加图像token
        if not use_original_query:
            image_token = VLM_IMAGE_TOKENS[QWEN2_VL]
            if image_token and image_token not in query_text:
                query_text = f"{image_token} {query_text}"
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  警告: 无法加载图像 {image_path}: {str(e)}")
            image = None
    
    inputs = processor(
        text=query_text,
        images=image,  # 根据开关决定是否传入图片
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 验证输入序列长度是否大于0
    if 'input_ids' in inputs:
        seq_len = inputs['input_ids'].shape[-1]
        if seq_len == 0:
            # 如果序列长度为0，也跳过该样本
            return None
    
    # 如果没有图片，这些字段可能不存在，需要检查
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    if 'image_grid_thw' in inputs:
        inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
    
    try:
        with torch.no_grad():
            qry_output = model(qry=inputs)["qry_reps"]
            
            # 打印第一条推理的输入和中间输出
            if not _first_inference_printed:
                print("\n" + "="*80)
                print("第一条推理的输入和输出")
                print("="*80)
                print(f"\n【输入信息】")
                print(f"  image_path: {image_path}")
                print(f"  subject: {subject_obj['class_name']} (bbox: {subject_obj['bbox']})")
                print(f"  object: {object_obj['class_name']} (bbox: {object_obj['bbox']})")
                print(f"  image_size: {original_width}x{original_height}")
                print(f"\n【CoT描述文本】")
                print(f"  {query_text[:500]}..." if len(query_text) > 500 else f"  {query_text}")
                print(f"\n【输入tensor信息】")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {value}")
                print(f"\n【中间输出（qry_output）】")
                print(f"  qry_output shape: {qry_output.shape}")
                print(f"  qry_output dtype: {qry_output.dtype}")
                print(f"  qry_output sample (first 10 values): {qry_output[0, :10].cpu().tolist()}")
                print("="*80 + "\n")
    except RuntimeError as e:
        # 捕获Flash Attention运行时错误
        if "FlashAttention only supports Ampere" in str(e):
            raise RuntimeError(
                "检测到Flash Attention运行时错误：您的GPU不支持Flash Attention。\n"
                "请在运行脚本前设置环境变量: export USE_FLASH_ATTENTION=0\n"
                f"原始错误: {str(e)}"
            )
        else:
            raise
    
    # 计算与所有谓词的相似度
    predicate_scores = []
    
    if predicate_vectors is not None:
        # 使用预计算的谓词向量（逐个计算相似度）
        with torch.no_grad():
            # qry_output: [1, hidden_dim]
            # predicate_vectors: [num_predicates, hidden_dim]
            for i, predicate in enumerate(PREDICATES):
                similarity = model.compute_similarity(
                    qry_output, 
                    predicate_vectors[i:i+1]  # 取单个谓词向量 [1, hidden_dim]
                )
                predicate_scores.append({
                    'predicate': predicate,
                    'similarity': similarity.item()
                })
    else:
        # 原始方法：逐个编码谓词（保持向后兼容）
        for predicate in PREDICATES:
            inputs = processor(text=predicate, images=None, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                tgt_output = model(tgt=inputs)["tgt_reps"]
                similarity = model.compute_similarity(qry_output, tgt_output)
            
            predicate_scores.append({
                'predicate': predicate,
                'similarity': similarity.item()
            })
    
    # 打印第一条推理的最终输出（predicate_scores）
    if not _first_inference_printed:
        print("\n" + "="*80)
        print("第一条推理的最终输出（Top-10谓词）")
        print("="*80)
        sorted_scores = sorted(predicate_scores, key=lambda x: x['similarity'], reverse=True)
        for i, score in enumerate(sorted_scores[:10], 1):
            print(f"  {i:2d}. {score['predicate']:20s}: {score['similarity']:.6f}")
        print("="*80 + "\n")
        _first_inference_printed = True  # 设置标志，避免后续重复打印
    
    return predicate_scores


def calculate_recall_at_k_per_image(image_candidate_predictions, k=50):
    """
    计算单张图片的recall@k
    现在支持所有物体两两配对的预测结果
    
    修改：先过滤no relation，再取top-k（与evaluate_results.py对齐）
    """
    # 获取该图片中所有GT关系（只统计relation_idx >= 0的，排除-1）
    gt_relations = set()
    for pred in image_candidate_predictions:
        if pred['has_gt'] and pred['relation_idx'] >= 0:
            gt_relations.add(pred['relation_idx'])
    
    # 第一步：过滤掉no relation的预测（从所有候选中）
    non_bg_candidates = []
    for pred in image_candidate_predictions:
        if pred.get('predicted_predicate') != 'no relation':
            non_bg_candidates.append(pred)
    
    # 第二步：按相似度排序，取top-k
    predictions_sorted = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
    actual_k = min(k, len(predictions_sorted))  # 如果候选数不足k，取全部
    top_k_predictions = predictions_sorted[:actual_k]
    
    # 第三步：在top-k中，只对GT关系对进行评估，统计召回的关系（去重）
    recalled_relations = set()
    for pred in top_k_predictions:
        # 只统计GT关系对中预测正确的
        if pred['relation_idx'] in gt_relations and pred['is_correct']:
            recalled_relations.add(pred['relation_idx'])
    
    # 总GT关系数
    total_gt_relations = len(gt_relations)
    
    recall = len(recalled_relations) / total_gt_relations if total_gt_relations > 0 else 0.0
    
    # 统计总预测对数（包括无GT的配对）
    total_pairs = len(set((pred['subject'], pred['object']) for pred in image_candidate_predictions))
    gt_pairs = len(set((pred['subject'], pred['object']) for pred in image_candidate_predictions if pred['has_gt']))
    
    return {
        'recall@k': recall,
        'k': k,
        'actual_k': actual_k,  # 实际取的数量（过滤no relation后）
        'recalled_relations': len(recalled_relations),
        'total_gt_relations': total_gt_relations,
        'total_candidates': len(image_candidate_predictions),
        'top_k_candidates': len(top_k_predictions),
        'total_pairs': total_pairs,  # 总预测对数
        'gt_pairs': gt_pairs,  # 有GT的配对对数
        'non_gt_pairs': total_pairs - gt_pairs  # 无GT的配对对数
    }


def calculate_mean_recall_per_predicate(per_image_candidates, predicates, k=50):
    """
    计算每个谓词类别的mean recall@50
    
    Args:
        per_image_candidates: dict, key为image_id, value为该图片的所有候选预测列表
        predicates: 所有谓词类别列表
        k: top-k
    
    Returns:
        dict: 每个谓词的recall和整体mean recall
    
    修改：先过滤no relation，再取top-k（与evaluate_results.py对齐）
    """
    # 初始化每个谓词的统计
    predicate_stats = {pred: {'hit': 0, 'total': 0} for pred in predicates}
    
    for image_id, candidates in per_image_candidates.items():
        # 获取该图片中所有GT关系（只统计relation_idx >= 0的，排除-1）
        gt_relations = set()
        for cand in candidates:
            if cand['relation_idx'] >= 0:
                gt_relations.add(cand['relation_idx'])
        
        # 第一步：过滤掉no relation的预测（从所有候选中）
        non_bg_candidates = []
        for cand in candidates:
            if cand.get('predicted_predicate') != 'no relation':
                non_bg_candidates.append(cand)
        
        # 第二步：按相似度排序，取top-k
        predictions_sorted = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
        actual_k = min(k, len(predictions_sorted))
        top_k_predictions = predictions_sorted[:actual_k]
        
        # 统计该图片中每个谓词类别的GT
        gt_predicates_in_image = {}
        recalled_predicates_in_image = {}
        
        for cand in candidates:
            if not cand['has_gt'] or cand['relation_idx'] == -1:
                continue  # 跳过没有GT的配对
                
            gt_pred = cand['gt_predicate']
            relation_idx = cand['relation_idx']
            
            # 统计GT（每个关系只算一次）
            if relation_idx not in gt_predicates_in_image:
                gt_predicates_in_image[relation_idx] = gt_pred
                predicate_stats[gt_pred]['total'] += 1
        
        # 第三步：在top-k中，只对GT关系对进行评估，统计召回的谓词
        for cand in top_k_predictions:
            # 只统计GT关系对中预测正确的
            if cand['relation_idx'] in gt_relations and cand['is_correct']:
                relation_idx = cand['relation_idx']
                gt_pred = cand['gt_predicate']
                
                # 每个关系只算一次召回
                if relation_idx not in recalled_predicates_in_image:
                    recalled_predicates_in_image[relation_idx] = gt_pred
                    predicate_stats[gt_pred]['hit'] += 1
    
    # 计算每个谓词的recall
    per_predicate_recall = {}
    valid_predicates = []
    
    for pred in predicates:
        total = predicate_stats[pred]['total']
        hit = predicate_stats[pred]['hit']
        
        if total > 0:
            recall = hit / total
            per_predicate_recall[pred] = {
                'recall': recall,
                'hit': hit,
                'total': total
            }
            valid_predicates.append(recall)
        else:
            per_predicate_recall[pred] = {
                'recall': 0.0,
                'hit': 0,
                'total': 0
            }
    
    # 计算mean recall（只对有GT的类别计算）
    mean_recall = sum(valid_predicates) / len(valid_predicates) if valid_predicates else 0.0
    
    return {
        'mean_recall@k': mean_recall,
        'k': k,
        'per_predicate_recall': per_predicate_recall,
        'num_valid_predicates': len(valid_predicates),
        'total_predicates': len(predicates)
    }


def calculate_average_recall_at_k(per_image_candidates, k=50):

    per_image_results = []
    total_recall = 0.0
    valid_images = 0
    
    for image_id, candidates in per_image_candidates.items():
        # 计算该图片的recall
        img_result = calculate_recall_at_k_per_image(candidates, k)
        img_result['image_id'] = image_id
        per_image_results.append(img_result)
        
        total_recall += img_result['recall@k']
        valid_images += 1
    
    # 计算平均recall
    avg_recall = total_recall / valid_images if valid_images > 0 else 0.0
    
    # 统计总体信息
    total_gt_relations = sum(r['total_gt_relations'] for r in per_image_results)
    total_recalled_relations = sum(r['recalled_relations'] for r in per_image_results)
    
    # 统计候选数不足k的图片数量
    images_with_insufficient_candidates = sum(1 for r in per_image_results if r['actual_k'] < k)
    
    return {
        'avg_recall@k': avg_recall,
        'k': k,
        'total_images': valid_images,
        'total_gt_relations': total_gt_relations,
        'total_recalled_relations': total_recalled_relations,
        'images_with_insufficient_candidates': images_with_insufficient_candidates,
        'per_image_results': per_image_results
    }


def process_data_shard(gpu_id, data_shard, model_args, data_args, predicate_vectors_dict, cot_map, result_queue, progress_queue, use_original_query=False, use_image=False):
    """
    在指定GPU上处理数据分片
    
    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        data_shard: 该GPU要处理的数据分片（图片列表）
        model_args: 模型参数
        data_args: 数据参数
        predicate_vectors_dict: 共享的谓词向量字典（通过Manager创建）
        cot_map: CoT描述映射字典 {(image_id, subject, object): cot_description}
        result_queue: 结果队列
        progress_queue: 进度队列
        use_original_query: 是否在cot_description前加上原始query
        use_image: 是否调用图像
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    try:
        # 加载处理器和模型（每个进程独立加载）
        processor = load_processor(model_args, data_args)
        
        # 尝试加载模型
        try:
            model = MMEBModel.load(model_args, is_trainable=False)
            model = model.to(device, dtype=torch.bfloat16)
            model.eval()
        except Exception as e:
            error_msg = str(e)
            if ("flash" in error_msg.lower() or 
                "ampere" in error_msg.lower() or 
                "attention" in error_msg.lower() and "support" in error_msg.lower()):
                # 强制使用eager模式
                os.environ["ATTN_IMPLEMENTATION"] = "eager"
                os.environ["USE_FLASH_ATTENTION"] = "0"
                
                import importlib
                import src.model.model
                importlib.reload(src.model.model)
                from src.model.model import MMEBModel as MMEBModelReloaded
                
                processor = load_processor(model_args, data_args)
                model = MMEBModelReloaded.load(model_args, is_trainable=False)
                model = model.to(device, dtype=torch.bfloat16)
                model.eval()
            else:
                raise
        
        # 获取或预计算谓词向量
        if gpu_id not in predicate_vectors_dict:
            # 如果该GPU还没有谓词向量，则预计算
            predicate_vectors = precompute_predicate_vectors(
                model, processor, PREDICATES, device=device,
                progress_queue=progress_queue, gpu_id=gpu_id
            )
            predicate_vectors_dict[gpu_id] = predicate_vectors.cpu()  # 保存到CPU以便共享
        else:
            # 使用共享的谓词向量（需要移回GPU）
            predicate_vectors = predicate_vectors_dict[gpu_id].to(device)
            progress_queue.put((gpu_id, f"GPU{gpu_id}: 使用共享谓词向量，开始处理图片..."))
        
        # 处理该GPU的数据分片
        per_image_candidates = {}
        all_relations_info = []
        processed_images = 0
        total_images = len(data_shard)
        missing_cot_count = 0
        total_pairs_checked = 0  # 统计检查的配对总数
        
        # 发送开始处理的消息
        progress_queue.put((gpu_id, f"GPU{gpu_id}: 开始处理 {total_images} 张图片"))
        
        for img_idx, img_data in enumerate(data_shard):
            image_id = img_data['image_id']
            image_path = img_data['image_path']
            objects = img_data['objects']
            relations = img_data['relations']
            
            # 检查图像是否存在
            if not os.path.exists(image_path):
                progress_queue.put((gpu_id, f"⚠️  GPU{gpu_id}: 图像不存在 {image_path}"))
                continue
            
            # 获取图像尺寸
            with Image.open(image_path) as img:
                original_width, original_height = img.size
            
            # 创建物体ID到物体信息的映射
            obj_dict = {obj['id']: obj for obj in objects}
            
            # 初始化该图片的候选列表
            image_candidates = []
            image_relation_idx = 0
            
            # 创建GT关系映射
            gt_relations_map = {}
            for relation in relations:
                subject_id = relation['subject_id']
                object_id = relation['object_id']
                gt_predicate = relation['predicate']
                if (subject_id, object_id) not in gt_relations_map:
                    gt_relations_map[(subject_id, object_id)] = []
                gt_relations_map[(subject_id, object_id)].append(gt_predicate)
            
            # 对所有物体进行两两配对预测
            object_ids = list(obj_dict.keys())
            for i, subject_id in enumerate(object_ids):
                for j, object_id in enumerate(object_ids):
                    if i == j:
                        continue
                    
                    subject_obj = obj_dict[subject_id]
                    object_obj = obj_dict[object_id]
                    
                    # 查找对应的CoT描述（优先使用物体ID）
                    subject_name = subject_obj['class_name'].strip()  # 去除可能的空格
                    object_name = object_obj['class_name'].strip()  # 去除可能的空格
                    total_pairs_checked += 1
                    
                    # 尝试多种key格式以确保匹配（优先使用物体ID）
                    cot_description = None
                    key_formats = []
                    
                    # 优先尝试使用物体ID的key格式
                    key_formats.extend([
                        (str(image_id), subject_id, object_id),  # 字符串image_id + 物体ID
                        (image_id, subject_id, object_id),  # 原始image_id格式 + 物体ID
                    ])
                    if isinstance(image_id, str) and image_id.isdigit():
                        key_formats.append((int(image_id), subject_id, object_id))
                    
                    # 向后兼容：如果没有找到，尝试使用类别名的key格式
                    key_formats.extend([
                        (str(image_id), subject_name, object_name),  # 字符串image_id + 类别名
                        (image_id, subject_name, object_name),  # 原始image_id格式 + 类别名
                    ])
                    if isinstance(image_id, str) and image_id.isdigit():
                        key_formats.append((int(image_id), subject_name, object_name))
                    
                    for key_format in key_formats:
                        if key_format and key_format in cot_map:
                            cot_description = cot_map[key_format]
                            break
                    
                    # 如果没有找到CoT描述，统计并跳过该样本
                    if not cot_description or not cot_description.strip():
                        missing_cot_count += 1
                        continue  # 跳过该配对
                    
                    # 预测关系（使用CoT描述）
                    predicate_scores = predict_relation(
                        model, processor, image_path,
                        subject_obj, object_obj,
                        original_width, original_height,
                        cot_description=cot_description,
                        predicate_vectors=predicate_vectors,
                        device=device,
                        use_original_query=use_original_query,
                        use_image=use_image
                    )
                    
                    # 如果predict_relation返回None（CoT描述为空或处理失败），跳过该样本
                    if predicate_scores is None:
                        missing_cot_count += 1
                        continue
                    
                    # 判断该配对是否有GT关系
                    has_gt = (subject_id, object_id) in gt_relations_map
                    gt_predicates = gt_relations_map.get((subject_id, object_id), [])
                    
                    # 记录关系信息
                    if has_gt:
                        for gt_predicate in gt_predicates:
                            all_relations_info.append({
                                'relation_idx': -1,  # 将在主进程重新分配
                                'image_id': image_id,
                                'image_relation_idx': image_relation_idx,
                                'subject_id': subject_id,  # 添加subject_id以区分同名物体
                                'object_id': object_id,  # 添加object_id以区分同名物体
                                'subject': subject_obj['class_name'],
                                'object': object_obj['class_name'],
                                'gt_predicate': gt_predicate
                            })
                            image_relation_idx += 1
                    
                    # 将该配对的50个谓词候选加入候选池
                    # 计算该配对对应的关系索引起始值
                    relation_idx_start = image_relation_idx - len(gt_predicates) if has_gt else -1
                    
                    for pred_score in predicate_scores:
                        is_correct = False
                        if has_gt and pred_score['predicate'] in gt_predicates:
                            is_correct = True
                        
                        # 如果预测正确，找到对应的关系索引
                        relation_idx = -1
                        if is_correct and has_gt:
                            # 找到该谓词在gt_predicates中的位置
                            for idx, gt_pred in enumerate(gt_predicates):
                                if gt_pred == pred_score['predicate']:
                                    relation_idx = relation_idx_start + idx
                                    break
                        
                        image_candidates.append({
                            'relation_idx': relation_idx,
                            'global_relation_idx': -1,  # 将在主进程重新分配
                            'image_id': image_id,
                            'subject_id': subject_id,  # 添加subject_id以区分同名物体
                            'object_id': object_id,  # 添加object_id以区分同名物体
                            'subject': subject_obj['class_name'],
                            'object': object_obj['class_name'],
                            'gt_predicate': gt_predicates[0] if gt_predicates else None,
                            'gt_predicates': gt_predicates,
                            'predicted_predicate': pred_score['predicate'],
                            'similarity': pred_score['similarity'],
                            'is_correct': is_correct,
                            'has_gt': has_gt
                        })
            
            per_image_candidates[image_id] = image_candidates
            processed_images += 1
            
            # 更新进度（每5张图片更新一次，平衡实时性和性能）
            if processed_images % 5 == 0 or processed_images == total_images:
                progress_queue.put((gpu_id, f"GPU{gpu_id}: 已处理 {processed_images}/{total_images} 张图片 ({processed_images*100//total_images}%)"))
        
        matched_count = total_pairs_checked - missing_cot_count
        match_rate = (matched_count / total_pairs_checked * 100) if total_pairs_checked > 0 else 0
        if missing_cot_count > 0:
            progress_queue.put((gpu_id, f"⚠️  GPU{gpu_id}: 警告：{missing_cot_count}/{total_pairs_checked} 个配对因CoT描述为空而被跳过（匹配率: {match_rate:.1f}%）"))
        else:
            progress_queue.put((gpu_id, f"✅ GPU{gpu_id}: 所有 {total_pairs_checked} 个配对都成功匹配到CoT描述"))
        
        # 将结果放入队列
        result_queue.put({
            'gpu_id': gpu_id,
            'per_image_candidates': per_image_candidates,
            'all_relations_info': all_relations_info,
            'missing_cot_count': missing_cot_count,  # 传递跳过的样本数量
            'total_pairs_checked': total_pairs_checked  # 传递检查的配对总数
        })
        
        progress_queue.put((gpu_id, f"✅ GPU{gpu_id}: 完成处理 {processed_images} 张图片"))
        
    except Exception as e:
        import traceback
        error_msg = f"GPU{gpu_id}处理失败: {str(e)}\n{traceback.format_exc()}"
        result_queue.put({
            'gpu_id': gpu_id,
            'error': error_msg
        })
        progress_queue.put((gpu_id, f"❌ {error_msg}"))



def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stage3: 使用Stage2的CoT数据进行场景图关系预测与Per-Image Recall@50计算')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='指定使用的GPU数量（默认：使用所有可用GPU，或从NUM_GPUS环境变量/配置变量读取）')
    parser.add_argument('--input_file', type=str, default=None,
                        help='输入文件路径（默认：使用配置文件中的INPUT_FILE）')
    parser.add_argument('--stage2_file', type=str, default=None,
                        help='Stage2输出文件路径（默认：使用配置文件中的STAGE2_OUTPUT_FILE）')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径（默认：使用配置文件中的OUTPUT_FILE）')
    parser.add_argument('--use_original_query', action='store_true', default=True,
                        help='是否在cot_description前加上原始query（默认：False）')
    parser.add_argument('--use_image', action='store_true', default=True,
                        help='是否调用图像（默认：False）')
    args = parser.parse_args()
    
    # 确定使用的GPU数量（优先级：命令行参数 > 环境变量 > 配置变量 > 所有GPU）
    num_gpus_to_use = args.num_gpus
    if num_gpus_to_use is None:
        num_gpus_to_use = os.environ.get('NUM_GPUS')
        if num_gpus_to_use is not None:
            num_gpus_to_use = int(num_gpus_to_use)
        else:
            num_gpus_to_use = NUM_GPUS
    
    # 确定输入输出文件
    input_file = args.input_file if args.input_file else INPUT_FILE
    stage2_file = args.stage2_file if args.stage2_file else STAGE2_OUTPUT_FILE
    output_file = args.output_file if args.output_file else OUTPUT_FILE
    
    print("="*80)
    print("Stage3: 使用Stage2的CoT数据进行场景图关系预测与Per-Image Recall@50计算")
    print("="*80)
    print(f"\n配置选项:")
    print(f"  使用原始query: {'是' if args.use_original_query else '否'}")
    print(f"  使用图像: {'是' if args.use_image else '否'}")
    print("="*80)

    # 检测可用GPU数量
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到CUDA设备")
        return
    
    total_gpus = torch.cuda.device_count()
    print(f"\n🔍 检测到 {total_gpus} 个GPU设备")
    for i in range(total_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"   GPU {i}: {gpu_name}")
    
    # 确定实际使用的GPU数量
    if num_gpus_to_use is None:
        num_gpus = total_gpus
        print(f"\n✅ 使用所有 {num_gpus} 个GPU")
    else:
        num_gpus = min(num_gpus_to_use, total_gpus)
        if num_gpus_to_use > total_gpus:
            print(f"\n⚠️  警告: 请求使用 {num_gpus_to_use} 个GPU，但只有 {total_gpus} 个可用，将使用 {num_gpus} 个GPU")
        else:
            print(f"\n✅ 使用指定的 {num_gpus} 个GPU (GPU 0-{num_gpus-1})")
    
    # 加载Stage2的CoT数据
    cot_map = load_stage2_cot_data(stage2_file)
    
    # 加载数据
    print(f"\n📖 正在加载数据: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_images = len(data)
    total_relations = sum(len(img['relations']) for img in data)
    print(f"   加载了 {total_images} 张图片，共 {total_relations} 个关系")
    
    # 准备模型参数
    model_args = ModelArguments(
        model_name='/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct',
        # checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct',
        checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/VLM2Vec-Qwen2VL-2B',
        # checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/train_5k_balance',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    
    data_args = DataArguments(
        resize_min_pixels=56 * 56,
        resize_max_pixels=28 * 28 * 1280
    )
    
    # 数据分片：将数据均匀分配到各个GPU
    print(f"\n📊 数据分片: 将 {total_images} 张图片分配到 {num_gpus} 个GPU")
    data_shards = []
    images_per_gpu = math.ceil(total_images / num_gpus)
    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        end_idx = min((i + 1) * images_per_gpu, total_images)
        shard = data[start_idx:end_idx]
        data_shards.append(shard)
        print(f"   GPU {i}: {len(shard)} 张图片 (索引 {start_idx}-{end_idx-1})")
    
    # 使用多进程进行多GPU并行推理
    print(f"\n🚀 开始多GPU并行推理...\n")
    
    # 创建共享字典和队列
    manager = Manager()
    predicate_vectors_dict = manager.dict()  # 共享的谓词向量字典
    result_queue = Queue()  # 结果队列
    progress_queue = Queue()  # 进度队列
    
    # 将cot_map转换为普通dict（multiprocessing需要可序列化的对象）
    # 注意：Manager().dict()不支持嵌套dict，所以直接传递普通dict
    # 由于cot_map是只读的，可以在每个进程中直接使用
    
    # 启动多个进程
    processes = []
    for gpu_id in range(num_gpus):
        if len(data_shards[gpu_id]) > 0:  # 只启动有数据的GPU进程
            p = Process(
                target=process_data_shard,
                args=(gpu_id, data_shards[gpu_id], model_args, data_args, 
                      predicate_vectors_dict, cot_map, result_queue, progress_queue,
                      args.use_original_query, args.use_image)
            )
            p.start()
            processes.append(p)
            print(f"   ✅ 启动GPU {gpu_id}进程")
    
    # 监控进度
    completed_gpus = set()
    progress_messages = {}
    
    def print_progress():
        """打印进度信息"""
        updated = False
        while not progress_queue.empty():
            try:
                gpu_id, message = progress_queue.get_nowait()
                progress_messages[gpu_id] = message
                updated = True
            except:
                break
        
        # 如果有更新，打印所有GPU的最新进度
        if updated:
            # 清屏并打印所有GPU的进度（使用\r实现覆盖）
            for gpu_id in range(num_gpus):
                if gpu_id in progress_messages:
                    print(f"   {progress_messages[gpu_id]}", flush=True)
    
    # 等待所有进程完成并收集结果
    print("\n📈 推理进度:")
    all_results = {}
    import time
    last_print_time = time.time()
    print_interval = 0.5  # 每0.5秒打印一次进度
    
    while len(completed_gpus) < len(processes):
        current_time = time.time()
        
        # 定期打印进度（每0.5秒或队列不为空时）
        if current_time - last_print_time >= print_interval or not progress_queue.empty():
            print_progress()
            last_print_time = current_time
        
        # 检查是否有新结果
        try:
            result = result_queue.get(timeout=0.1)  # 减少超时时间，更频繁检查进度
            if 'error' in result:
                print(f"\n❌ {result['error']}", flush=True)
                completed_gpus.add(result['gpu_id'])
            else:
                all_results[result['gpu_id']] = result
                completed_gpus.add(result['gpu_id'])
                print(f"\n   ✅ GPU {result['gpu_id']} 完成", flush=True)
        except:
            # 超时是正常的，继续循环检查进度
            pass
    
    # 最后打印一次所有GPU的最终进度
    print_progress()
    
    # 等待所有进程结束
    for p in processes:
        p.join()
        p.terminate()
    
    print("\n✅ 所有GPU处理完成，正在合并结果...")
    
    # 合并所有GPU的结果
    per_image_candidates = {}
    all_relations_info = []
    total_missing_cot_count = 0  # 统计总的跳过样本数量
    total_pairs_checked_all = 0  # 统计总的检查配对数量
    
    # 按GPU ID顺序合并结果
    for gpu_id in sorted(all_results.keys()):
        result = all_results[gpu_id]
        if 'error' in result:
            continue
        
        # 合并每张图片的候选
        for image_id, candidates in result['per_image_candidates'].items():
            per_image_candidates[image_id] = candidates
        
        # 累计跳过的样本数量和检查的配对数量
        if 'missing_cot_count' in result:
            total_missing_cot_count += result['missing_cot_count']
        if 'total_pairs_checked' in result:
            total_pairs_checked_all += result['total_pairs_checked']
    
    # 重新分配关系索引（基于合并后的数据）
    print("   重新分配关系索引...")
    global_relation_idx = 0
    
    # 按图片ID排序处理，确保一致性
    for image_id in sorted(per_image_candidates.keys()):
        candidates = per_image_candidates[image_id]
        image_relation_idx = 0
        
        # 收集该图片的所有GT关系（去重，使用物体ID区分同名物体）
        gt_relations_set = set()
        gt_relations_list = []
        for cand in candidates:
            if cand['has_gt'] and cand['gt_predicate']:
                # 使用物体ID作为唯一标识，以区分同名物体
                subject_id = cand.get('subject_id', None)
                object_id = cand.get('object_id', None)
                if subject_id is not None and object_id is not None:
                    key = (subject_id, object_id, cand['gt_predicate'])
                else:
                    # 向后兼容：如果没有物体ID，则使用类别名
                    key = (cand['subject'], cand['object'], cand['gt_predicate'])
                if key not in gt_relations_set:
                    gt_relations_set.add(key)
                    gt_relations_list.append({
                        'subject_id': subject_id,
                        'object_id': object_id,
                        'subject': cand['subject'],
                        'object': cand['object'],
                        'gt_predicate': cand['gt_predicate']
                    })
        
        # 为每个GT关系分配全局索引
        image_relation_idx_map = {}  # (subject_id, object_id, gt_predicate) 或 (subject, object, gt_predicate) -> relation_idx
        for rel_info in gt_relations_list:
            if rel_info['subject_id'] is not None and rel_info['object_id'] is not None:
                key = (rel_info['subject_id'], rel_info['object_id'], rel_info['gt_predicate'])
            else:
                key = (rel_info['subject'], rel_info['object'], rel_info['gt_predicate'])
            image_relation_idx_map[key] = global_relation_idx
            
            all_relations_info.append({
                'relation_idx': global_relation_idx,
                'image_id': image_id,
                'image_relation_idx': image_relation_idx,
                'subject_id': rel_info['subject_id'],
                'object_id': rel_info['object_id'],
                'subject': rel_info['subject'],
                'object': rel_info['object'],
                'gt_predicate': rel_info['gt_predicate']
            })
            global_relation_idx += 1
            image_relation_idx += 1
        
        # 更新候选中的关系索引
        for cand in candidates:
            if cand['has_gt'] and cand['gt_predicate']:
                # 使用物体ID进行匹配
                subject_id = cand.get('subject_id', None)
                object_id = cand.get('object_id', None)
                if subject_id is not None and object_id is not None:
                    key = (subject_id, object_id, cand['gt_predicate'])
                else:
                    # 向后兼容
                    key = (cand['subject'], cand['object'], cand['gt_predicate'])
                if key in image_relation_idx_map:
                    rel_idx = image_relation_idx_map[key]
                    cand['relation_idx'] = rel_idx
                    cand['global_relation_idx'] = rel_idx
                else:
                    cand['relation_idx'] = -1
                    cand['global_relation_idx'] = -1
            else:
                cand['relation_idx'] = -1
                cand['global_relation_idx'] = -1
    
    # 多GPU模式下，结果已经在process_data_shard中处理完成并合并
    # 现在直接进入结果统计和保存阶段
    
    print(f"\n✅ 预测完成！")
    print(f"   总图片数: {len(per_image_candidates)}")
    print(f"   总GT关系数: {len(all_relations_info)}")
    total_candidates = sum(len(candidates) for candidates in per_image_candidates.values())
    print(f"   总候选预测数: {total_candidates}")
    if total_pairs_checked_all > 0:
        matched_count_all = total_pairs_checked_all - total_missing_cot_count
        match_rate_all = (matched_count_all / total_pairs_checked_all * 100) if total_pairs_checked_all > 0 else 0
        print(f"   CoT匹配统计: {matched_count_all}/{total_pairs_checked_all} 个配对成功匹配（匹配率: {match_rate_all:.1f}%）")
        if total_missing_cot_count > 0:
            print(f"   ⚠️  跳过的样本数（CoT描述为空）: {total_missing_cot_count}")
    
    # 统计配对信息
    total_pairs = 0
    total_gt_pairs = 0
    for candidates in per_image_candidates.values():
        pairs_in_image = set((cand['subject'], cand['object']) for cand in candidates)
        gt_pairs_in_image = set((cand['subject'], cand['object']) for cand in candidates if cand['has_gt'])
        total_pairs += len(pairs_in_image)
        total_gt_pairs += len(gt_pairs_in_image)
    
    print(f"   总预测配对对数: {total_pairs}")
    print(f"   有GT的配对对数: {total_gt_pairs}")
    print(f"   无GT的配对对数: {total_pairs - total_gt_pairs}")
    
    # 5. 计算Per-Image Recall@50并取平均
    print("\n📊 计算Per-Image Recall@50（每张图片独立计算再平均）...")
    recall_results = calculate_average_recall_at_k(per_image_candidates, k=50)
    
    # 5.1 计算Mean Recall@50（针对每个谓词类别）
    print("\n📊 计算Mean Recall@50（针对所有谓词类别）...")
    mean_recall_results = calculate_mean_recall_per_predicate(per_image_candidates, PREDICATES, k=50)
    
    print("\n" + "="*80)
    print("评估结果 (Per-Image Recall@50)")
    print("="*80)
    print(f"平均 Recall@{recall_results['k']}: {recall_results['avg_recall@k']:.4f} ({recall_results['avg_recall@k']*100:.2f}%)")
    print(f"总图片数: {recall_results['total_images']}")
    print(f"总召回关系数: {recall_results['total_recalled_relations']}/{recall_results['total_gt_relations']}")
    
    # 计算平均配对统计
    if len(recall_results['per_image_results']) > 0:
        avg_total_pairs = sum(r.get('total_pairs', 0) for r in recall_results['per_image_results']) / len(recall_results['per_image_results'])
        avg_gt_pairs = sum(r.get('gt_pairs', 0) for r in recall_results['per_image_results']) / len(recall_results['per_image_results'])
        avg_non_gt_pairs = sum(r.get('non_gt_pairs', 0) for r in recall_results['per_image_results']) / len(recall_results['per_image_results'])
        
        print(f"平均每张图片预测配对对数: {avg_total_pairs:.1f}")
        print(f"平均每张图片有GT的配对对数: {avg_gt_pairs:.1f}")
        print(f"平均每张图片无GT的配对对数: {avg_non_gt_pairs:.1f}")
    else:
        print("⚠️  警告: 没有处理任何图片，无法计算平均配对统计")
        print(f"平均每张图片预测配对对数: 0.0")
        print(f"平均每张图片有GT的配对对数: 0.0")
        print(f"平均每张图片无GT的配对对数: 0.0")
    
    if recall_results['images_with_insufficient_candidates'] > 0:
        print(f"候选数不足{recall_results['k']}的图片: {recall_results['images_with_insufficient_candidates']}/{recall_results['total_images']}")
    print("="*80)
    
    print("\n" + "="*80)
    print("评估结果 (Mean Recall@50 - 所有谓词类别)")
    print("="*80)
    print(f"Mean Recall@{mean_recall_results['k']}: {mean_recall_results['mean_recall@k']:.4f} ({mean_recall_results['mean_recall@k']*100:.2f}%)")
    print(f"有效谓词类别数: {mean_recall_results['num_valid_predicates']}/{mean_recall_results['total_predicates']}")
    print("="*80)
    
    # 显示每个谓词的recall（前10个和后10个）
    print("\n谓词类别Recall详情（按recall排序）:")
    sorted_predicates = sorted(
        mean_recall_results['per_predicate_recall'].items(),
        key=lambda x: x[1]['recall'],
        reverse=True
    )
    
    # 只显示有GT的谓词
    predicates_with_gt = [(pred, stats) for pred, stats in sorted_predicates if stats['total'] > 0]
    
    if len(predicates_with_gt) > 0:
        print("\n  Top-10 表现最好的谓词:")
        for i, (pred, stats) in enumerate(predicates_with_gt[:10], 1):
            print(f"    {i:2d}. {pred:20s}: R={stats['recall']:.4f} ({stats['hit']:3d}/{stats['total']:3d})")
        
        if len(predicates_with_gt) > 10:
            print("\n  Bottom-10 表现最差的谓词:")
            for i, (pred, stats) in enumerate(predicates_with_gt[-10:], 1):
                print(f"    {i:2d}. {pred:20s}: R={stats['recall']:.4f} ({stats['hit']:3d}/{stats['total']:3d})")
    
    # 6. 显示每张图片的recall分布
    per_image_recalls = [r['recall@k'] for r in recall_results['per_image_results']]
    if per_image_recalls:
        print(f"\nRecall分布统计:")
        print(f"  最大值: {max(per_image_recalls):.4f}")
        print(f"  最小值: {min(per_image_recalls):.4f}")
        print(f"  中位数: {sorted(per_image_recalls)[len(per_image_recalls)//2]:.4f}")
    
    # 7. 收集所有候选用于展示（可选）
    all_candidate_predictions = []
    for candidates in per_image_candidates.values():
        all_candidate_predictions.extend(candidates)
    
    candidates_sorted = sorted(all_candidate_predictions, key=lambda x: x['similarity'], reverse=True)
    top50_global_candidates = candidates_sorted[:100]
    
    # 7.1 为每张图片收集Top-100候选结果
    print("\n📦 正在整理每张图片的Top-100候选结果...")
    per_image_top100_candidates = {}
    total_top100_candidates = 0
    
    for image_id, candidates in per_image_candidates.items():
        # 按相似度排序
        sorted_candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
        # 取Top-100
        top100 = sorted_candidates[:min(100, len(sorted_candidates))]
        per_image_top100_candidates[image_id] = top100
        total_top100_candidates += len(top100)
    
    print(f"   收集了 {len(per_image_top100_candidates)} 张图片的Top-100候选")
    print(f"   总候选数: {total_top100_candidates}")
    
    # 8. 保存结果
    print(f"\n💾 正在保存结果到: {output_file}")
    output_data = {
        'summary': {
            'evaluation_method': 'per-image-all-pairs-stage3',
            'stage2_file': stage2_file,
            'total_images': len(per_image_candidates),
            'total_gt_relations': len(all_relations_info),
            'total_candidates': total_candidates,
            'total_top100_candidates': total_top100_candidates,  # 新增：Top-100候选总数
            'avg_recall@50': recall_results['avg_recall@k'],
            'mean_recall@50': mean_recall_results['mean_recall@k'],
            'total_recalled_relations': recall_results['total_recalled_relations'],
            'total_gt_relations': recall_results['total_gt_relations'],
            'num_valid_predicates': mean_recall_results['num_valid_predicates'],
            'images_with_insufficient_candidates': recall_results['images_with_insufficient_candidates'],
            # 新增配对统计
            'total_pairs': total_pairs,
            'total_gt_pairs': total_gt_pairs,
            'total_non_gt_pairs': total_pairs - total_gt_pairs,
            'avg_pairs_per_image': total_pairs / len(per_image_candidates) if len(per_image_candidates) > 0 else 0,
            'avg_gt_pairs_per_image': total_gt_pairs / len(per_image_candidates) if len(per_image_candidates) > 0 else 0,
            # 跳过的样本统计
            'skipped_samples_missing_cot': total_missing_cot_count,
            # 配置选项
            'use_original_query': args.use_original_query,
            'use_image': args.use_image
        },
        'per_image_results': recall_results['per_image_results'],
        'mean_recall_per_predicate': mean_recall_results['per_predicate_recall'],
        'all_relations': all_relations_info,
        'per_image_top100_candidates': per_image_top100_candidates,  # 新增：每张图片的Top-100候选
        'top50_global_candidates': top50_global_candidates,  # 全局排序的top50（参考用）
        # 'all_candidates': all_candidate_predictions  # 完整的候选列表（可选，可能很大）
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("✅ 结果已保存！")
    
    # 9. 显示一些样例
    print("\n" + "="*80)
    print("Per-Image Recall样例（前5张图片）")
    print("="*80)
    for i, img_result in enumerate(recall_results['per_image_results'][:5], 1):
        print(f"\n{i}. 图片#{img_result['image_id']}")
        print(f"   Recall@50: {img_result['recall@k']:.4f} ({img_result['recall@k']*100:.2f}%)")
        print(f"   召回: {img_result['recalled_relations']}/{img_result['total_gt_relations']} 关系")
        actual_k_info = f" (实际取{img_result['actual_k']}个)" if img_result['actual_k'] < img_result['k'] else ""
        print(f"   候选数: {img_result['total_candidates']} (Top-{img_result['k']}中取{img_result['top_k_candidates']}个{actual_k_info})")
        print(f"   配对统计: 总配对{img_result.get('total_pairs', 0)}对, 有GT配对{img_result.get('gt_pairs', 0)}对, 无GT配对{img_result.get('non_gt_pairs', 0)}对")
    
    print("\n" + "="*80)
    print("全局Top-50候选预测样例（前10个，仅供参考）")
    print("="*80)
    for i, pred in enumerate(top50_global_candidates[:10], 1):
        status = "✅" if pred['is_correct'] else "❌"
        print(f"\n{i}. {status} 排名#{i} (相似度: {pred['similarity']:.4f})")
        print(f"   图片#{pred['image_id']}, 关系#{pred['relation_idx']}: {pred['subject']} --[{pred['predicted_predicate']}]--> {pred['object']}")
        print(f"   GT谓词: {pred['gt_predicate']}")


if __name__ == "__main__":
    # 设置multiprocessing启动方法为'spawn'，以支持CUDA多进程
    # Linux系统默认使用'fork'，但CUDA不支持在fork的子进程中重新初始化
    # 必须在导入multiprocessing后、创建任何进程之前设置
    try:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过启动方法，忽略错误
        pass
    
    main()

