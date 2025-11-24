
import json
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
import warnings


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


INPUT_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_200_images_cot.json"
OUTPUT_FILE = "/public/home/xiaojw2025/Workspace/VLM2Vec/cot/predict/recall_results_200_cot.json"

# 50个谓词列表
PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind",
    "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
    "in", "in front of", "laying on", "looking at", "lying on", "made of",
    "mounted on", "near", "of", "on", "on back of", "over", "painted on",
    "parked on", "part of", "playing", "riding", "says", "sitting on",
    "standing on", "to", "under", "using", "walking in", "walking on",
    "watching", "wearing", "wears", "with","no relation"
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


def precompute_predicate_vectors(model, processor, predicates):
    """
    预计算所有谓词的向量表示（只需要计算一次）
    
    Args:
        model: VLM2Vec模型
        processor: 文本处理器
        predicates: 谓词列表
    
    Returns:
        predicate_vectors: [num_predicates, hidden_dim] 的tensor
    """
    print("🔧 预计算谓词向量...")
    predicate_vectors = []
    
    for predicate in tqdm(predicates, desc="编码谓词"):
        predicate_text = f"The subject is {predicate} the object."
        inputs = processor(text=predicate_text, images=None, return_tensors="pt")
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        
        with torch.no_grad():
            tgt_output = model(tgt=inputs)["tgt_reps"]
            predicate_vectors.append(tgt_output)
    
    # 堆叠成一个tensor: [num_predicates, hidden_dim]
    predicate_vectors = torch.cat(predicate_vectors, dim=0)
    print(f"✅ 谓词向量预计算完成，shape: {predicate_vectors.shape}")
    
    return predicate_vectors


def predict_relation(model, processor, query_text, predicate_vectors=None):
    """
    预测关系，使用预计算的谓词向量
    
    Args:
        model: VLM2Vec模型
        processor: 文本处理器
        query_text: 查询文本（已经包含关系描述的文本）
        predicate_vectors: 预计算的谓词向量 [num_predicates, hidden_dim]，如果为None则实时计算
    
    Returns:
        predicate_scores: 谓词及其相似度分数的列表
    """
    # 使用query_text作为query（纯文本，不需要图像）
    # query_text已经是描述两个物体关系的文本了
    inputs = processor(text=query_text, images=None, return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    
    try:
        with torch.no_grad():
            qry_output = model(tgt=inputs)["tgt_reps"]  # 使用tgt因为这是纯文本输入
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
            pred_inputs = processor(text=f"The subject is {predicate} the object.", images=None, return_tensors="pt")
            pred_inputs = {key: value.to('cuda') for key, value in pred_inputs.items()}
            
            with torch.no_grad():
                tgt_output = model(tgt=pred_inputs)["tgt_reps"]
                similarity = model.compute_similarity(qry_output, tgt_output)
            
            predicate_scores.append({
                'predicate': predicate,
                'similarity': similarity.item()
            })
    
    return predicate_scores


def calculate_recall_at_k_per_image(image_candidate_predictions, k=50):
    """
    计算单张图片的recall@k
    现在支持所有物体两两配对的预测结果
    """
    # 按相似度排序，取top-k（如果候选数不足k，则取全部）
    predictions_sorted = sorted(image_candidate_predictions, key=lambda x: x['similarity'], reverse=True)
    actual_k = min(k, len(predictions_sorted))  # 如果候选数不足k，取全部
    top_k_predictions = predictions_sorted[:actual_k]
    
    # 统计top-k中预测正确的关系（去重，每个关系只算一次），排除no relation预测
    recalled_relations = set()
    for pred in top_k_predictions:
        if pred['is_correct'] and pred['relation_idx'] != -1 and pred.get('predicted_predicate') != 'no relation':  # 只统计有GT的关系，排除no relation预测
            recalled_relations.add(pred['relation_idx'])
    
    # 总GT关系数（从候选中提取有GT的关系的唯一relation_idx数量）
    gt_relations = [pred for pred in image_candidate_predictions if pred['has_gt'] and pred['relation_idx'] != -1]
    total_gt_relations = len(set(pred['relation_idx'] for pred in gt_relations))
    
    recall = len(recalled_relations) / total_gt_relations if total_gt_relations > 0 else 0.0
    
    # 统计总预测对数（包括无GT的配对）
    total_pairs = len(set((pred['subject'], pred['object']) for pred in image_candidate_predictions))
    gt_pairs = len(set((pred['subject'], pred['object']) for pred in image_candidate_predictions if pred['has_gt']))
    
    return {
        'recall@k': recall,
        'k': k,
        'actual_k': actual_k,  # 实际取的数量
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
    """
    # 初始化每个谓词的统计
    predicate_stats = {pred: {'hit': 0, 'total': 0} for pred in predicates}
    
    for image_id, candidates in per_image_candidates.items():
        # 按相似度排序，取top-k（如果候选数不足k，则取全部）
        predictions_sorted = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
        actual_k = min(k, len(predictions_sorted))
        top_k_predictions = predictions_sorted[:actual_k]
        
        # 统计每个谓词类别的GT数量和召回数量
        # 先统计该图片中每个谓词类别的GT
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
        
        # 统计Top-K中召回的谓词（每个关系只算一次），排除no relation预测
        for cand in top_k_predictions:
            if cand['is_correct'] and cand.get('predicted_predicate') != 'no relation':
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



def main():
    print("="*80)
    print("场景图关系预测与Per-Image Recall@50计算")
    print("="*80)

    # 加载数据
    print(f"\n📖 正在加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data_json = json.load(f)
    
    # 适配新数据格式：从detailed_results中提取数据
    if isinstance(data_json, dict) and 'detailed_results' in data_json:
        data = data_json['detailed_results']
        print(f"   检测到新数据格式，从detailed_results加载")
    else:
        data = data_json
    
    total_images = len(data)
    total_relations = sum(len(img['relations']) for img in data)
    print(f"   加载了 {total_images} 张图片，共 {total_relations} 个关系")
    
    #  加载模型
    print("\n🔧 正在加载VLM2Vec模型...")
    

    model_args = ModelArguments(
        model_name='/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct',
        # checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/VLM2Vec-Qwen2VL-2B',
        checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/final',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    
    data_args = DataArguments(
        resize_min_pixels=56 * 56,
        resize_max_pixels=28 * 28 * 1280
    )
    
    processor = load_processor(model_args, data_args)
    
    # 尝试加载模型，如果flash attention失败则降级
    try:
        model = MMEBModel.load(model_args)
        model = model.to('cuda', dtype=torch.bfloat16)
        model.eval()
        print("   ✅ 模型加载完成")
    except Exception as e:
        error_msg = str(e)
        # 检查是否是Flash Attention相关错误
        if ("flash" in error_msg.lower() or 
            "ampere" in error_msg.lower() or 
            "attention" in error_msg.lower() and "support" in error_msg.lower()):
            print(f"\n⚠️  模型加载/运行失败: {error_msg[:200]}")
            print("   检测到Flash Attention兼容性问题")
            print("   尝试降级到eager模式...")
            
            # 强制使用eager模式（通过环境变量）
            os.environ["ATTN_IMPLEMENTATION"] = "eager"
            os.environ["USE_FLASH_ATTENTION"] = "0"
            
            # 需要重新导入模块以应用新的环境变量
            import importlib
            import src.model.model
            importlib.reload(src.model.model)
            from src.model.model import MMEBModel as MMEBModelReloaded
            
            try:
                # 重新加载处理器和模型
                processor = load_processor(model_args, data_args)
                model = MMEBModelReloaded.load(model_args)
                model = model.to('cuda', dtype=torch.bfloat16)
                model.eval()
                print("   ✅ 模型加载完成 (使用eager模式)")
            except Exception as e2:
                print(f"\n❌ 降级后仍然失败: {e2}")
                raise
        else:
            print(f"\n❌ 模型加载失败: {error_msg}")
            raise
    
    # 3. 预计算谓词向量（只需要一次）
    print("\n🚀 预计算谓词向量（加速推理）...\n")
    predicate_vectors = precompute_predicate_vectors(model, processor, PREDICATES)
    
    # 4. 批量预测
    print("\n🚀 开始批量预测...\n")
    
    per_image_candidates = {}  # 按图片组织的候选预测 {image_id: [candidates]}
    all_relations_info = []  # 每个关系的详细信息
    
    global_relation_idx = 0  # 全局关系索引
    
    for img_idx, img_data in enumerate(tqdm(data, desc="处理图片")):
        image_id = img_data['image_id']
        image_path = img_data['image_path']
        relations = img_data['relations']  # 新格式中，relations已经包含所有配对
        
        # 初始化该图片的候选列表
        image_candidates = []
        image_relation_idx = 0  # 该图片内的关系索引
        
        # 创建GT关系映射，用于判断预测是否正确
        # 注意：新格式中可能没有GT predicate，只有prediction描述
        # 如果需要GT，可能需要从原始数据中读取，或者假设每个relation都有一个GT
        gt_relations_map = {}
        for relation in relations:
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            # 新格式可能没有predicate字段，只有prediction描述
            # 如果有predicate字段则使用，否则跳过GT检查
            if 'predicate' in relation:
                gt_predicate = relation['predicate']
                if (subject_id, object_id) not in gt_relations_map:
                    gt_relations_map[(subject_id, object_id)] = []
                gt_relations_map[(subject_id, object_id)].append(gt_predicate)
        
        # 遍历relations中的每个关系配对
        for relation in relations:
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            subject_name = relation['subject_name']
            object_name = relation['object_name']
            query_text = relation['prediction']  # 使用prediction作为query
            
            # 检查是否有GT predicate
            has_gt = (subject_id, object_id) in gt_relations_map
            gt_predicates = gt_relations_map.get((subject_id, object_id), [])
            
            # 使用prediction作为query，预测50个谓词的相似度
            try:
                predicate_scores = predict_relation(
                    model, processor, query_text,
                    predicate_vectors=predicate_vectors  # 传入预计算的向量
                )
            except Exception as e:
                print(f"⚠️  预测失败 (image {image_id}, relation {subject_id}-{object_id}): {e}")
                continue
            
            # 记录该关系的信息（如果有GT）
            if has_gt:
                for gt_predicate in gt_predicates:
                    all_relations_info.append({
                        'relation_idx': global_relation_idx,
                        'image_id': image_id,
                        'image_relation_idx': image_relation_idx,
                        'subject': subject_name,
                        'object': object_name,
                        'gt_predicate': gt_predicate
                    })
                    image_relation_idx += 1
                    global_relation_idx += 1
            
            # 将该配对的50个谓词候选加入该图片的候选池
            for pred_score in predicate_scores:
                # 判断预测是否正确（如果该配对有GT关系）
                is_correct = False
                if has_gt and pred_score['predicate'] in gt_predicates:
                    is_correct = True
                
                image_candidates.append({
                    'relation_idx': image_relation_idx - len(gt_predicates) if has_gt else -1,
                    'global_relation_idx': global_relation_idx - len(gt_predicates) if has_gt else -1,
                    'image_id': image_id,
                    'subject': subject_name,
                    'object': object_name,
                    'gt_predicate': gt_predicates[0] if gt_predicates else None,
                    'gt_predicates': gt_predicates,
                    'predicted_predicate': pred_score['predicate'],
                    'similarity': pred_score['similarity'],
                    'is_correct': is_correct,
                    'has_gt': has_gt,
                    'query_text': query_text  # 保存原始query文本
                })
        
        # 保存该图片的所有候选
        per_image_candidates[image_id] = image_candidates
    
    print(f"\n✅ 预测完成！")
    print(f"   总图片数: {len(per_image_candidates)}")
    print(f"   总GT关系数: {len(all_relations_info)}")
    total_candidates = sum(len(candidates) for candidates in per_image_candidates.values())
    print(f"   总候选预测数: {total_candidates}")
    
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
    avg_total_pairs = sum(r.get('total_pairs', 0) for r in recall_results['per_image_results']) / len(recall_results['per_image_results'])
    avg_gt_pairs = sum(r.get('gt_pairs', 0) for r in recall_results['per_image_results']) / len(recall_results['per_image_results'])
    avg_non_gt_pairs = sum(r.get('non_gt_pairs', 0) for r in recall_results['per_image_results']) / len(recall_results['per_image_results'])
    
    print(f"平均每张图片预测配对对数: {avg_total_pairs:.1f}")
    print(f"平均每张图片有GT的配对对数: {avg_gt_pairs:.1f}")
    print(f"平均每张图片无GT的配对对数: {avg_non_gt_pairs:.1f}")
    
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
    print(f"\n💾 正在保存结果到: {OUTPUT_FILE}")
    output_data = {
        'summary': {
            'evaluation_method': 'per-image-all-pairs',
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
            'avg_gt_pairs_per_image': total_gt_pairs / len(per_image_candidates) if len(per_image_candidates) > 0 else 0
        },
        'per_image_results': recall_results['per_image_results'],
        'mean_recall_per_predicate': mean_recall_results['per_predicate_recall'],
        'all_relations': all_relations_info,
        'per_image_top100_candidates': per_image_top100_candidates,  # 新增：每张图片的Top-100候选
        'top50_global_candidates': top50_global_candidates,  # 全局排序的top50（参考用）
        # 'all_candidates': all_candidate_predictions  # 完整的候选列表（可选，可能很大）
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
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
    main()

