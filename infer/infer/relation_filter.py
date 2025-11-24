
import json
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
import warnings
from pathlib import Path
torch.cuda.empty_cache() 
# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent  # embedding/infer/
project_root = current_dir.parent.parent  # VLM2Vec/
sys.path.insert(0, str(project_root))


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


INPUT_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_2000_images.json"
OUTPUT_FILE = "/public/home/xiaojw2025/Workspace/VLM2Vec/predict/recall_results_2000_mmmeb_filter.json"

# 关系判断类别（二分类：有关系 vs 无关系）
RELATION_CATEGORIES = ["has_relation", "no_relation"]


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


def precompute_relation_vectors(model, processor):
    """
    预计算关系类别向量（二分类：有关系 vs 无关系）
    
    Args:
        model: VLM2Vec模型
        processor: 文本处理器
    
    Returns:
        relation_vectors: dict, {'has_relation': tensor, 'no_relation': tensor}
    """
    print("🔧 预计算关系类别向量...")
    relation_vectors = {}
    
    # 有关系的描述
    has_relation_text = "The subject and object have a relationship."
    inputs = processor(text=has_relation_text, images=None, return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        relation_vectors['has_relation'] = model(tgt=inputs)["tgt_reps"]
    
    # 无关系的描述
    no_relation_text = "The subject and object have no relationship."
    inputs = processor(text=no_relation_text, images=None, return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        relation_vectors['no_relation'] = model(tgt=inputs)["tgt_reps"]
    
    print(f"✅ 关系类别向量预计算完成")
    print(f"   has_relation shape: {relation_vectors['has_relation'].shape}")
    print(f"   no_relation shape: {relation_vectors['no_relation'].shape}")
    
    return relation_vectors


def predict_relation_binary(model, processor, image_path, subject_obj, object_obj, 
                            original_width, original_height, relation_vectors=None):
    """
    二分类预测：判断两个物体之间是否有关系
    
    Args:
        relation_vectors: 预计算的关系向量 dict {'has_relation': tensor, 'no_relation': tensor}
    
    Returns:
        dict: {'has_relation': similarity_score, 'no_relation': similarity_score, 'predicted_category': str}
    """
    # 构建subject和object的特殊token
    subj_bbox_token = format_bbox_as_special_token(
        subject_obj['bbox'], True, original_width, original_height
    )
    obj_bbox_token = format_bbox_as_special_token(
        object_obj['bbox'], True, original_width, original_height
    )
    subj_ref = format_object_with_ref(subject_obj['class_name'])
    obj_ref = format_object_with_ref(object_obj['class_name'])
    
    # 修改查询文本，聚焦于判断是否有关系
    query_text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]} In the given image, the subject {subj_ref} is located at {subj_bbox_token}, the object {obj_ref} is located at {obj_bbox_token}. Do they have any relationship?"
    
    inputs = processor(
        text=query_text,
        images=Image.open(image_path),
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
    
    try:
        with torch.no_grad():
            qry_output = model(qry=inputs)["qry_reps"]
    except RuntimeError as e:
        if "FlashAttention only supports Ampere" in str(e):
            raise RuntimeError(
                "检测到Flash Attention运行时错误：您的GPU不支持Flash Attention。\n"
                "请在运行脚本前设置环境变量: export USE_FLASH_ATTENTION=0\n"
                f"原始错误: {str(e)}"
            )
        else:
            raise
    
    # 计算与两个关系类别的相似度
    scores = {}
    
    if relation_vectors is not None:
        # 使用预计算的关系向量
        with torch.no_grad():
            for category in RELATION_CATEGORIES:
                similarity = model.compute_similarity(
                    qry_output, 
                    relation_vectors[category]
                )
                scores[category] = similarity.item()
    else:
        # 实时计算（保持向后兼容）
        for category in RELATION_CATEGORIES:
            if category == "has_relation":
                text = "The subject and object have a relationship."
            else:
                text = "The subject and object have no relationship."
            
            inputs = processor(text=text, images=None, return_tensors="pt")
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            
            with torch.no_grad():
                tgt_output = model(tgt=inputs)["tgt_reps"]
                similarity = model.compute_similarity(qry_output, tgt_output)
                scores[category] = similarity.item()
    
    # 判断预测类别（相似度更高的）
    predicted_category = max(scores, key=scores.get)
    
    return {
        'has_relation_similarity': scores['has_relation'],
        'no_relation_similarity': scores['no_relation'],
        'predicted_category': predicted_category,
        'confidence': scores[predicted_category]
    }


def calculate_binary_classification_metrics(image_pair_predictions):
    """
    计算二分类（有关系 vs 无关系）的评估指标
    
    Args:
        image_pair_predictions: list of dicts, 每个dict包含一个物体对的预测结果
    
    Returns:
        dict: 包含准确率、精确率、召回率、F1等指标
    """
    true_positives = 0  # 正确预测有关系
    false_positives = 0  # 错误预测有关系（实际无关系）
    true_negatives = 0  # 正确预测无关系
    false_negatives = 0  # 错误预测无关系（实际有关系）
    
    total_pairs = len(image_pair_predictions)
    gt_pairs_count = 0  # 实际有关系的配对数
    
    for pred in image_pair_predictions:
        has_gt = pred['has_gt']  # Ground Truth: 是否有关系
        predicted_has_relation = (pred['predicted_category'] == 'has_relation')
        
        if has_gt:
            gt_pairs_count += 1
            if predicted_has_relation:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_has_relation:
                false_positives += 1
            else:
                true_negatives += 1
    
    # 计算各项指标
    accuracy = (true_positives + true_negatives) / total_pairs if total_pairs > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'total_pairs': total_pairs,
        'gt_pairs': gt_pairs_count,
        'non_gt_pairs': total_pairs - gt_pairs_count
    }


def calculate_per_image_metrics(per_image_pairs):
    """
    计算每张图片的二分类指标并汇总
    
    Args:
        per_image_pairs: dict, key为image_id, value为该图片的物体对预测列表
    
    Returns:
        dict: 包含总体和各图片的评估指标
    """
    per_image_results = []
    
    overall_tp = 0
    overall_fp = 0
    overall_tn = 0
    overall_fn = 0
    
    for image_id, pairs in per_image_pairs.items():
        # 计算该图片的指标
        metrics = calculate_binary_classification_metrics(pairs)
        metrics['image_id'] = image_id
        per_image_results.append(metrics)
        
        # 累积总体统计
        overall_tp += metrics['true_positives']
        overall_fp += metrics['false_positives']
        overall_tn += metrics['true_negatives']
        overall_fn += metrics['false_negatives']
    
    # 计算总体指标
    total_pairs = overall_tp + overall_fp + overall_tn + overall_fn
    overall_accuracy = (overall_tp + overall_tn) / total_pairs if total_pairs > 0 else 0.0
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # 计算平均图片级别指标
    avg_accuracy = sum(r['accuracy'] for r in per_image_results) / len(per_image_results) if per_image_results else 0.0
    avg_precision = sum(r['precision'] for r in per_image_results) / len(per_image_results) if per_image_results else 0.0
    avg_recall = sum(r['recall'] for r in per_image_results) / len(per_image_results) if per_image_results else 0.0
    avg_f1 = sum(r['f1_score'] for r in per_image_results) / len(per_image_results) if per_image_results else 0.0
    
    return {
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': overall_tp,
            'false_positives': overall_fp,
            'true_negatives': overall_tn,
            'false_negatives': overall_fn,
            'total_pairs': total_pairs
        },
        'average_per_image_metrics': {
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        },
        'total_images': len(per_image_results),
        'per_image_results': per_image_results
    }



def main():
    print("="*80)
    print("场景图关系二分类预测（判断物体对是否有关系）")
    print("="*80)

    # 加载数据
    print(f"\n📖 正在加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    total_images = len(data)
    total_relations = sum(len(img['relations']) for img in data)
    print(f"   加载了 {total_images} 张图片，共 {total_relations} 个关系")
    
    #  加载模型
    print("\n🔧 正在加载VLM2Vec模型...")
    

    model_args = ModelArguments(
        model_name='/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct',
        # checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct',
        checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/VLM2Vec-Qwen2VL-2B',
        # checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/final',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True  # 使用 LoRA 模型
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
    
    # 3. 预计算关系类别向量（只需要一次）
    print("\n🚀 预计算关系类别向量（加速推理）...\n")
    relation_vectors = precompute_relation_vectors(model, processor)
    
    # 4. 批量预测（二分类：判断是否有关系）
    print("\n🚀 开始批量预测...\n")
    
    per_image_pairs = {}  # 按图片组织的物体对预测 {image_id: [pair_predictions]}
    
    for img_idx, img_data in enumerate(tqdm(data, desc="处理图片")):
        image_id = img_data['image_id']
        image_path = img_data['image_path']
        objects = img_data['objects']
        relations = img_data['relations']
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"⚠️  警告: 图像不存在 {image_path}")
            continue
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            original_width, original_height = img.size
        
        # 创建物体ID到物体信息的映射
        obj_dict = {obj['id']: obj for obj in objects}
        
        # 创建GT关系映射，用于判断是否有关系（不关心具体谓词）
        gt_relations_set = set()
        for relation in relations:
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            # 只记录有关系的配对，不关心具体谓词
            gt_relations_set.add((subject_id, object_id))
        
        # 对所有物体进行两两配对预测
        image_pair_predictions = []
        object_ids = list(obj_dict.keys())
        
        for i, subject_id in enumerate(object_ids):
            for j, object_id in enumerate(object_ids):
                # 跳过自己与自己配对
                if i == j:
                    continue
                
                subject_obj = obj_dict[subject_id]
                object_obj = obj_dict[object_id]
                
                # 二分类预测：判断是否有关系
                prediction = predict_relation_binary(
                    model, processor, image_path,
                    subject_obj, object_obj,
                    original_width, original_height,
                    relation_vectors=relation_vectors
                )
                
                # 判断该配对是否有GT关系（不关心具体谓词）
                has_gt = (subject_id, object_id) in gt_relations_set
                
                # 保存该物体对的预测结果
                image_pair_predictions.append({
                    'image_id': image_id,
                    'subject_id': subject_id,
                    'object_id': object_id,
                    'subject': subject_obj['class_name'],
                    'object': object_obj['class_name'],
                    'has_gt': has_gt,  # Ground Truth: 是否有关系
                    'predicted_category': prediction['predicted_category'],  # has_relation 或 no_relation
                    'has_relation_similarity': prediction['has_relation_similarity'],
                    'no_relation_similarity': prediction['no_relation_similarity'],
                    'confidence': prediction['confidence']
                })
        
        # 保存该图片的所有物体对预测
        per_image_pairs[image_id] = image_pair_predictions
    
    print(f"\n✅ 预测完成！")
    print(f"   总图片数: {len(per_image_pairs)}")
    
    # 统计配对信息
    total_pairs = sum(len(pairs) for pairs in per_image_pairs.values())
    total_gt_pairs = sum(sum(1 for p in pairs if p['has_gt']) for pairs in per_image_pairs.values())
    
    print(f"   总物体对数: {total_pairs}")
    print(f"   有GT关系的配对数: {total_gt_pairs}")
    print(f"   无GT关系的配对数: {total_pairs - total_gt_pairs}")
    
    # 5. 计算二分类评估指标
    print("\n📊 计算二分类评估指标...")
    metrics_results = calculate_per_image_metrics(per_image_pairs)
    
    print("\n" + "="*80)
    print("评估结果 - 总体指标（所有物体对）")
    print("="*80)
    overall = metrics_results['overall_metrics']
    print(f"准确率 (Accuracy):  {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
    print(f"精确率 (Precision): {overall['precision']:.4f} ({overall['precision']*100:.2f}%)")
    print(f"召回率 (Recall):    {overall['recall']:.4f} ({overall['recall']*100:.2f}%)")
    print(f"F1分数 (F1-Score):  {overall['f1_score']:.4f} ({overall['f1_score']*100:.2f}%)")
    print(f"\n混淆矩阵:")
    print(f"  真正例 (TP): {overall['true_positives']:6d}  (正确识别有关系)")
    print(f"  假正例 (FP): {overall['false_positives']:6d}  (错误识别有关系)")
    print(f"  真负例 (TN): {overall['true_negatives']:6d}  (正确识别无关系)")
    print(f"  假负例 (FN): {overall['false_negatives']:6d}  (错误识别无关系)")
    print(f"  总配对数:    {overall['total_pairs']:6d}")
    print("="*80)
    
    print("\n" + "="*80)
    print("评估结果 - 平均图片级别指标")
    print("="*80)
    avg = metrics_results['average_per_image_metrics']
    print(f"平均准确率:  {avg['accuracy']:.4f} ({avg['accuracy']*100:.2f}%)")
    print(f"平均精确率:  {avg['precision']:.4f} ({avg['precision']*100:.2f}%)")
    print(f"平均召回率:  {avg['recall']:.4f} ({avg['recall']*100:.2f}%)")
    print(f"平均F1分数:  {avg['f1_score']:.4f} ({avg['f1_score']*100:.2f}%)")
    print(f"总图片数:    {metrics_results['total_images']}")
    print("="*80)
    
    # 6. 保存结果
    print(f"\n💾 正在保存结果到: {OUTPUT_FILE}")
    output_data = {
        'summary': {
            'evaluation_method': 'binary_classification',  # 二分类方法
            'task': 'relation_detection',  # 关系检测（不是谓词分类）
            'total_images': len(per_image_pairs),
            'total_pairs': total_pairs,
            'total_gt_pairs': total_gt_pairs,
            'total_non_gt_pairs': total_pairs - total_gt_pairs,
            'avg_pairs_per_image': total_pairs / len(per_image_pairs) if len(per_image_pairs) > 0 else 0,
            'avg_gt_pairs_per_image': total_gt_pairs / len(per_image_pairs) if len(per_image_pairs) > 0 else 0
        },
        'overall_metrics': metrics_results['overall_metrics'],
        'average_per_image_metrics': metrics_results['average_per_image_metrics'],
        'per_image_results': metrics_results['per_image_results'],
        # 注意：不保存所有物体对的预测细节（太大），只保存汇总统计
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("✅ 结果已保存！")
    
    # 7. 显示一些样例图片
    print("\n" + "="*80)
    print("图片级别评估样例（前5张图片）")
    print("="*80)
    for i, img_result in enumerate(metrics_results['per_image_results'][:5], 1):
        print(f"\n{i}. 图片 #{img_result['image_id']}")
        print(f"   准确率: {img_result['accuracy']:.4f}, 精确率: {img_result['precision']:.4f}, "
              f"召回率: {img_result['recall']:.4f}, F1: {img_result['f1_score']:.4f}")
        print(f"   TP: {img_result['true_positives']}, FP: {img_result['false_positives']}, "
              f"TN: {img_result['true_negatives']}, FN: {img_result['false_negatives']}")
        print(f"   总配对: {img_result['total_pairs']}, 有GT: {img_result['gt_pairs']}, "
              f"无GT: {img_result['non_gt_pairs']}")
    
    # 8. 显示一些预测样例
    print("\n" + "="*80)
    print("预测样例（前10对，按置信度排序）")
    print("="*80)
    
    # 收集所有预测，按置信度排序
    all_predictions = []
    for pairs in per_image_pairs.values():
        all_predictions.extend(pairs)
    
    # 按置信度排序
    sorted_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
    
    for i, pred in enumerate(sorted_predictions[:10], 1):
        gt_label = "有关系" if pred['has_gt'] else "无关系"
        pred_label = "有关系" if pred['predicted_category'] == 'has_relation' else "无关系"
        status = "✅" if (pred['has_gt'] == (pred['predicted_category'] == 'has_relation')) else "❌"
        
        print(f"\n{i}. {status} 图片#{pred['image_id']}: {pred['subject']} <-> {pred['object']}")
        print(f"   GT: {gt_label} | 预测: {pred_label} (置信度: {pred['confidence']:.4f})")
        print(f"   有关系相似度: {pred['has_relation_similarity']:.4f}")
        print(f"   无关系相似度: {pred['no_relation_similarity']:.4f}")


if __name__ == "__main__":
    main()

