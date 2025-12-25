"""
基于完整GT数据的评估脚本
从原始GT文件中读取所有GT pair，基于完整的GT pair计算召回率
"""

import json
import argparse
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set

# Base和Novel类谓词分类映射
PREDICATE_CATEGORY_MAPPING = {
    "above": "base", "across": "novel", "against": "base", "along": "novel", "and": "novel",
    "at": "base", "attached to": "base", "behind": "base", "belonging to": "base", "between": "base",
    "carrying": "base", "covered in": "base", "covering": "base", "eating": "novel", "flying in": "novel",
    "for": "base", "from": "base", "growing on": "novel", "hanging from": "base", "has": "base",
    "holding": "base", "in": "base", "in front of": "base", "laying on": "novel", "looking at": "base",
    "lying on": "novel", "made of": "base", "mounted on": "novel", "near": "base", "of": "base",
    "on": "base", "on back of": "novel", "over": "base", "painted on": "novel", "parked on": "base",
    "part of": "novel", "playing": "base", "riding": "base", "says": "novel", "sitting on": "base",
    "standing on": "base", "to": "base", "under": "base", "using": "novel", "walking in": "novel",
    "walking on": "base", "watching": "base", "wearing": "base", "wears": "base", "with": "base",
    "no relation": "base"  # 添加no relation，虽然通常不参与统计
}


def load_gt_data(gt_file: str) -> Dict:
    """
    从GT文件中加载所有GT pair
    
    Args:
        gt_file: GT文件路径
        
    Returns:
        字典，key为image_id，value为该图片的所有GT pair集合
        GT pair格式: (subject_id, object_id, predicate) 用于区分同名物体
        同时保存类别名映射: {image_id: {subject_id: class_name, ...}}
    """
    print(f"📖 正在加载GT文件: {gt_file}")
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_pairs_per_image = {}
    object_id_to_class = {}  # {image_id: {object_id: class_name}}
    
    for item in gt_data:
        image_id = item['image_id']
        objects = {obj['id']: obj['class_name'] for obj in item['objects']}
        relations = item['relations']
        object_id_to_class[image_id] = objects
        
        # 构建该图片的所有GT pair，使用物体ID区分同名物体
        gt_pairs = set()
        for rel in relations:
            subject_id = rel['subject_id']
            object_id = rel['object_id']
            predicate = rel['predicate']
            
            if subject_id in objects and object_id in objects:
                # 使用 (subject_id, object_id, predicate) 作为唯一标识，以区分同名物体
                gt_pairs.add((subject_id, object_id, predicate))
        
        gt_pairs_per_image[image_id] = gt_pairs
    
    print(f"✅ GT文件加载完成，共 {len(gt_pairs_per_image)} 张图片")
    total_gt_pairs = sum(len(pairs) for pairs in gt_pairs_per_image.values())
    print(f"   总GT pair数: {total_gt_pairs}\n")
    
    # 返回GT pairs和物体ID到类别名的映射
    return gt_pairs_per_image, object_id_to_class


def load_results(json_path: str) -> Dict:
    """加载预测结果JSON文件"""
    print(f"📖 正在加载结果文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 加载完成\n")
    return data


def get_per_image_candidates(data: Dict) -> Dict:
    """
    获取按图片分组的候选列表
    
    Returns:
        字典，key为image_id，value为该图片的所有候选列表
    """
    per_image_candidates = {}
    
    if 'per_image_top100_candidates' in data:
        print("   使用 per_image_top100_candidates 字段...")
        per_image_candidates = data['per_image_top100_candidates']
    elif 'all_candidates' in data:
        print("   使用 all_candidates 字段...")
        all_candidates = data['all_candidates']
        # 按 image_id 分组
        per_image_candidates_list = defaultdict(list)
        for cand in all_candidates:
            per_image_candidates_list[cand['image_id']].append(cand)
        per_image_candidates = dict(per_image_candidates_list)
    else:
        print("⚠️  JSON中没有保存候选列表 (per_image_top100_candidates 或 all_candidates 字段缺失)")
        return None
    
    return per_image_candidates


def calculate_recall_with_full_gt(
    data: Dict, 
    gt_pairs_per_image: Dict,
    object_id_to_class: Dict,
    k_values: List[int] = [50, 100]
) -> Dict:
    """
    基于完整GT数据计算召回率
    
    Args:
        data: 预测结果数据
        gt_pairs_per_image: 每张图片的完整GT pair集合
        k_values: K值列表
        
    Returns:
        包含召回率统计的字典
    """
    print(f"📊 基于完整GT数据计算Top-K召回率 (K={k_values})...")
    
    per_image_candidates = get_per_image_candidates(data)
    if per_image_candidates is None:
        return None
    
    results = {}
    
    for k in k_values:
        print(f"\n   计算 Recall@{k}...")
        
        total_gt_pairs = 0  # 所有GT中的pair总数（完整GT）
        total_recalled_pairs = 0  # 在top-k中被召回的pair数
        total_gt_pairs_in_candidates = 0  # 在候选列表中的GT pair数
        total_recalled_pairs_in_candidates = 0  # 在候选列表中且被召回的pair数
        
        # Base和Novel类分别统计
        total_gt_pairs_base = 0
        total_recalled_pairs_base = 0
        total_gt_pairs_novel = 0
        total_recalled_pairs_novel = 0
        
        # 用于统计每张图片的情况
        image_recalls = []
        image_recalls_in_candidates = []  # 基于候选列表中的GT pair的召回率
        image_recalls_base = []  # Base类召回率
        image_recalls_novel = []  # Novel类召回率
        
        for image_id_str, candidates in per_image_candidates.items():
            # 统一image_id类型（转换为整数）
            try:
                image_id = int(image_id_str)
            except (ValueError, TypeError):
                image_id = image_id_str
            
            # 获取该图片的完整GT pair集合
            full_gt_pairs = gt_pairs_per_image.get(image_id, set())
            
            if len(full_gt_pairs) == 0:
                continue
            
            # 获取该图片的物体ID到类别名映射
            image_object_map = object_id_to_class.get(image_id, {})
            
            # 构建候选列表中的GT pair集合（用于对比）
            gt_pairs_in_candidates = set()
            for cand in candidates:
                has_gt = cand.get('has_gt', False)
                if has_gt and cand.get('relation_idx', -1) >= 0:
                    # 优先使用物体ID，如果没有则使用类别名
                    subject_id = cand.get('subject_id', None)
                    object_id = cand.get('object_id', None)
                    predicate = cand.get('gt_predicate', '')
                    if subject_id is not None and object_id is not None and predicate:
                        gt_pairs_in_candidates.add((subject_id, object_id, predicate))
                    else:
                        # 向后兼容：使用类别名
                        subject = cand.get('subject', '')
                        object_name = cand.get('object', '')
                        if subject and object_name and predicate:
                            gt_pairs_in_candidates.add((subject, object_name, predicate))
            
            # 过滤掉no relation的预测
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 按相似度排序，取top-k
            sorted_candidates = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            top_k = sorted_candidates[:min(k, len(sorted_candidates))]
            
            # 统计在top-k中确实存在于完整GT中的pair
            recalled_pairs = set()
            for cand in top_k:
                # 优先使用物体ID进行匹配
                subject_id = cand.get('subject_id', None)
                object_id = cand.get('object_id', None)
                predicate = cand.get('predicted_predicate', '')
                
                if subject_id is not None and object_id is not None and predicate:
                    pair_key = (subject_id, object_id, predicate)
                    if pair_key in full_gt_pairs:
                        recalled_pairs.add(pair_key)
                else:
                    # 向后兼容：如果没有物体ID，尝试使用类别名匹配
                    # 需要将类别名转换为物体ID（可能不准确，因为可能有多个同名物体）
                    subject = cand.get('subject', '')
                    object_name = cand.get('object', '')
                    if subject and object_name and predicate:
                        # 尝试找到匹配的物体ID对
                        found_match = False
                        for gt_subj_id, gt_obj_id, gt_pred in full_gt_pairs:
                            if (gt_pred == predicate and 
                                image_object_map.get(gt_subj_id) == subject and 
                                image_object_map.get(gt_obj_id) == object_name):
                                recalled_pairs.add((gt_subj_id, gt_obj_id, predicate))
                                found_match = True
                                break
            
            # 分别统计Base和Novel类的GT pairs
            gt_pairs_base = set()
            gt_pairs_novel = set()
            for pair in full_gt_pairs:
                predicate = pair[2]  # pair格式: (subject_id, object_id, predicate)
                category = PREDICATE_CATEGORY_MAPPING.get(predicate, "base")  # 默认base
                if category == "base":
                    gt_pairs_base.add(pair)
                elif category == "novel":
                    gt_pairs_novel.add(pair)
            
            # 分别统计Base和Novel类的召回
            recalled_pairs_base = recalled_pairs & gt_pairs_base
            recalled_pairs_novel = recalled_pairs & gt_pairs_novel
            
            # 计算基于完整GT的召回率
            recalled_count = len(recalled_pairs)
            gt_count = len(full_gt_pairs)
            recall = recalled_count / gt_count if gt_count > 0 else 0.0
            
            # 计算Base和Novel类的召回率
            gt_count_base = len(gt_pairs_base)
            recalled_count_base = len(recalled_pairs_base)
            recall_base = recalled_count_base / gt_count_base if gt_count_base > 0 else 0.0
            
            gt_count_novel = len(gt_pairs_novel)
            recalled_count_novel = len(recalled_pairs_novel)
            recall_novel = recalled_count_novel / gt_count_novel if gt_count_novel > 0 else 0.0
            
            image_recalls.append(recall)
            if gt_count_base > 0:
                image_recalls_base.append(recall_base)
            if gt_count_novel > 0:
                image_recalls_novel.append(recall_novel)
            
            total_gt_pairs += gt_count
            total_recalled_pairs += recalled_count
            total_gt_pairs_base += gt_count_base
            total_recalled_pairs_base += recalled_count_base
            total_gt_pairs_novel += gt_count_novel
            total_recalled_pairs_novel += recalled_count_novel
            
            # 计算基于候选列表中GT pair的召回率（用于对比）
            recalled_in_candidates = len(recalled_pairs & gt_pairs_in_candidates)
            gt_in_candidates_count = len(gt_pairs_in_candidates)
            recall_in_candidates = recalled_in_candidates / gt_in_candidates_count if gt_in_candidates_count > 0 else 0.0
            
            image_recalls_in_candidates.append(recall_in_candidates)
            total_gt_pairs_in_candidates += gt_in_candidates_count
            total_recalled_pairs_in_candidates += recalled_in_candidates
        
        # 计算平均召回率和整体召回率
        avg_recall = np.mean(image_recalls) if image_recalls else 0.0
        overall_recall = total_recalled_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
        
        # Base和Novel类的平均召回率和整体召回率
        avg_recall_base = np.mean(image_recalls_base) if image_recalls_base else 0.0
        overall_recall_base = total_recalled_pairs_base / total_gt_pairs_base if total_gt_pairs_base > 0 else 0.0
        avg_recall_novel = np.mean(image_recalls_novel) if image_recalls_novel else 0.0
        overall_recall_novel = total_recalled_pairs_novel / total_gt_pairs_novel if total_gt_pairs_novel > 0 else 0.0
        
        # 基于候选列表中的GT pair的召回率（用于对比）
        avg_recall_in_candidates = np.mean(image_recalls_in_candidates) if image_recalls_in_candidates else 0.0
        overall_recall_in_candidates = total_recalled_pairs_in_candidates / total_gt_pairs_in_candidates if total_gt_pairs_in_candidates > 0 else 0.0
        
        # Stage1覆盖率：有多少GT pair进入了候选列表
        stage1_coverage = total_gt_pairs_in_candidates / total_gt_pairs if total_gt_pairs > 0 else 0.0
        
        results[f'recall@{k}'] = {
            'avg_recall': avg_recall,
            'overall_recall': overall_recall,
            'total_gt_pairs': total_gt_pairs,  # 完整GT pair数
            'total_recalled_pairs': total_recalled_pairs,
            'num_images': len(image_recalls),
            'image_recalls': image_recalls,
            # Base和Novel类统计
            'avg_recall_base': avg_recall_base,
            'overall_recall_base': overall_recall_base,
            'total_gt_pairs_base': total_gt_pairs_base,
            'total_recalled_pairs_base': total_recalled_pairs_base,
            'avg_recall_novel': avg_recall_novel,
            'overall_recall_novel': overall_recall_novel,
            'total_gt_pairs_novel': total_gt_pairs_novel,
            'total_recalled_pairs_novel': total_recalled_pairs_novel,
            # 对比指标：基于候选列表中的GT pair
            'avg_recall_in_candidates': avg_recall_in_candidates,
            'overall_recall_in_candidates': overall_recall_in_candidates,
            'total_gt_pairs_in_candidates': total_gt_pairs_in_candidates,
            'total_recalled_pairs_in_candidates': total_recalled_pairs_in_candidates,
            # Stage1覆盖率
            'stage1_coverage': stage1_coverage
        }
        
        print(f"   基于完整GT数据:")
        print(f"     平均召回率: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
        print(f"     整体召回率: {overall_recall:.4f} ({overall_recall*100:.2f}%)")
        print(f"     统计: {total_recalled_pairs}/{total_gt_pairs} pairs被召回，共{len(image_recalls)}张图片")
        print(f"   Base类谓词:")
        print(f"     平均召回率: {avg_recall_base:.4f} ({avg_recall_base*100:.2f}%)")
        print(f"     整体召回率: {overall_recall_base:.4f} ({overall_recall_base*100:.2f}%)")
        print(f"     统计: {total_recalled_pairs_base}/{total_gt_pairs_base} pairs被召回")
        print(f"   Novel类谓词:")
        print(f"     平均召回率: {avg_recall_novel:.4f} ({avg_recall_novel*100:.2f}%)")
        print(f"     整体召回率: {overall_recall_novel:.4f} ({overall_recall_novel*100:.2f}%)")
        print(f"     统计: {total_recalled_pairs_novel}/{total_gt_pairs_novel} pairs被召回")
        print(f"   基于候选列表中的GT pair (对比):")
        print(f"     平均召回率: {avg_recall_in_candidates:.4f} ({avg_recall_in_candidates*100:.2f}%)")
        print(f"     整体召回率: {overall_recall_in_candidates:.4f} ({overall_recall_in_candidates*100:.2f}%)")
        print(f"     统计: {total_recalled_pairs_in_candidates}/{total_gt_pairs_in_candidates} pairs被召回")
        print(f"   Stage1覆盖率: {stage1_coverage:.4f} ({stage1_coverage*100:.2f}%) - {total_gt_pairs_in_candidates}/{total_gt_pairs} GT pairs进入候选列表")
    
    print()
    return results


def calculate_confusion_matrix_with_full_gt(
    data: Dict,
    gt_pairs_per_image: Dict,
    object_id_to_class: Dict,
    k_values: List[int] = [50, 100]
) -> Dict:
    """
    基于完整GT数据计算混淆矩阵
    
    Args:
        data: 预测结果数据
        gt_pairs_per_image: 每张图片的完整GT pair集合
        k_values: K值列表
        
    Returns:
        包含混淆矩阵的字典
    """
    print(f"📊 基于完整GT数据计算混淆矩阵 (K={k_values})...")
    
    per_image_candidates = get_per_image_candidates(data)
    if per_image_candidates is None:
        return None
    
    results = {}
    
    for k in k_values:
        print(f"\n   计算 K={k} 的混淆矩阵...")
        
        tp = 0  # True Positive: 完整GT中的pair被召回了
        fn = 0  # False Negative: 完整GT中的pair没有被召回
        fp = 0  # False Positive: top-k中的pair不在完整GT中
        tn = 0  # True Negative: top-k中不在完整GT中的pair（relation_idx == -1）
        
        # 每张图片内的指标统计
        image_recalls = []
        image_precisions = []
        image_f1s = []
        
        for image_id_str, candidates in per_image_candidates.items():
            # 统一image_id类型（转换为整数）
            try:
                image_id = int(image_id_str)
            except (ValueError, TypeError):
                image_id = image_id_str
            
            # 获取该图片的完整GT pair集合
            full_gt_pairs = gt_pairs_per_image.get(image_id, set())
            
            if len(full_gt_pairs) == 0:
                continue
            
            # 获取该图片的物体ID到类别名映射
            image_object_map = object_id_to_class.get(image_id, {})
            
            # 过滤掉no relation的预测
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 按相似度排序，取top-k
            sorted_candidates = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            top_k = sorted_candidates[:min(k, len(sorted_candidates))]
            
            # 统计top-k中的pair（使用物体ID）
            top_k_pairs = set()
            top_k_non_gt_pairs = set()
            
            for cand in top_k:
                # 优先使用物体ID
                subject_id = cand.get('subject_id', None)
                object_id = cand.get('object_id', None)
                predicate = cand.get('predicted_predicate', '')
                
                if subject_id is not None and object_id is not None and predicate:
                    pair_key = (subject_id, object_id, predicate)
                    
                    relation_idx = cand.get('relation_idx', -1)
                    if relation_idx >= 0:  # 在候选列表的GT中
                        top_k_pairs.add(pair_key)
                    else:
                        top_k_non_gt_pairs.add(pair_key)
                else:
                    # 向后兼容：如果没有物体ID，尝试使用类别名
                    subject = cand.get('subject', '')
                    object_name = cand.get('object', '')
                    if subject and object_name and predicate:
                        # 尝试找到匹配的物体ID对
                        found_match = False
                        for gt_subj_id, gt_obj_id, gt_pred in full_gt_pairs:
                            if (gt_pred == predicate and 
                                image_object_map.get(gt_subj_id) == subject and 
                                image_object_map.get(gt_obj_id) == object_name):
                                pair_key = (gt_subj_id, gt_obj_id, predicate)
                                relation_idx = cand.get('relation_idx', -1)
                                if relation_idx >= 0:
                                    top_k_pairs.add(pair_key)
                                else:
                                    top_k_non_gt_pairs.add(pair_key)
                                found_match = True
                                break
                        if not found_match:
                            # 如果找不到匹配，使用类别名（可能不准确）
                            pair_key = (subject, object_name, predicate)
                            relation_idx = cand.get('relation_idx', -1)
                            if relation_idx >= 0:
                                top_k_pairs.add(pair_key)
                            else:
                                top_k_non_gt_pairs.add(pair_key)
            
            # 计算当前图片的TP、FN、FP、TN
            # TP: 完整GT中的pair在top-k中出现了
            image_tp = len(full_gt_pairs & top_k_pairs)
            # FN: 完整GT中的pair在top-k中没有出现
            image_fn = len(full_gt_pairs - top_k_pairs)
            # FP: top-k中的pair不在完整GT中
            image_fp = len(top_k_pairs - full_gt_pairs)
            # TN: top-k中不在完整GT中的pair
            image_tn = len(top_k_non_gt_pairs)
            
            # 累计到全局统计
            tp += image_tp
            fn += image_fn
            fp += image_fp
            tn += image_tn
            
            # 计算当前图片的指标
            image_total_gt = image_tp + image_fn
            
            image_recall = 0.0
            image_precision = 0.0
            
            if image_total_gt > 0:
                image_recall = image_tp / image_total_gt
                image_recalls.append(image_recall)
            
            image_total_topk = image_tp + image_fp
            if image_total_topk > 0:
                image_precision = image_tp / image_total_topk
                image_precisions.append(image_precision)
            
            # 计算F1分数
            if image_precision > 0 and image_recall > 0:
                image_f1 = 2 * image_precision * image_recall / (image_precision + image_recall)
                image_f1s.append(image_f1)
        
        # 构建2x2混淆矩阵
        cm = np.array([[tp, fn],
                       [fp, tn]])
        
        # 计算指标
        total_gt_pairs = tp + fn  # 完整GT中的pair总数
        total_top_k_pairs = tp + fp  # top-k中在完整GT中的pair总数
        
        # 整体指标
        recall_overall = tp / total_gt_pairs if total_gt_pairs > 0 else 0.0
        precision_overall = tp / total_top_k_pairs if total_top_k_pairs > 0 else 0.0
        f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall) if (precision_overall + recall_overall) > 0 else 0.0
        
        # 平均指标
        recall_avg = np.mean(image_recalls) if image_recalls else 0.0
        precision_avg = np.mean(image_precisions) if image_precisions else 0.0
        f1_avg = np.mean(image_f1s) if image_f1s else 0.0
        
        results[f'confusion_matrix@{k}'] = {
            'matrix': cm.tolist(),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_gt_pairs': int(total_gt_pairs),
            'total_top_k_pairs': int(total_top_k_pairs),
            'recall_overall': recall_overall,
            'precision_overall': precision_overall,
            'f1_overall': f1_overall,
            'recall_avg': recall_avg,
            'precision_avg': precision_avg,
            'f1_avg': f1_avg
        }
        
        print(f"   混淆矩阵 (2x2):")
        print(f"               预测为正例(在top-k)  预测为负例(不在top-k)")
        print(f"   实际为正例(在GT)    TP={tp:6d}        FN={fn:6d}")
        print(f"   实际为负例(不在GT)  FP={fp:6d}        TN={tn:6d}")
        print(f"   指标 (每张图片平均):")
        print(f"     召回率 (Recall): {recall_avg:.4f} ({recall_avg*100:.2f}%)")
        print(f"     精确率 (Precision): {precision_avg:.4f} ({precision_avg*100:.2f}%)")
        print(f"     F1分数: {f1_avg:.4f}")
        print(f"   指标 (整体累计):")
        print(f"     召回率 (Recall): {recall_overall:.4f} ({recall_overall*100:.2f}%)")
        print(f"     精确率 (Precision): {precision_overall:.4f} ({precision_overall*100:.2f}%)")
        print(f"     F1分数: {f1_overall:.4f}")
        print(f"   统计: 完整GT中有{total_gt_pairs}个pair, top-k中有{total_top_k_pairs}个GT pair")
    
    print()
    return results


def calculate_mean_recall_with_full_gt(
    data: Dict,
    gt_pairs_per_image: Dict,
    object_id_to_class: Dict,
    k_values: List[int] = [50, 100]
) -> Dict:
    """
    基于完整GT数据计算每个谓词类别的Mean Recall
    
    Args:
        data: 预测结果数据
        gt_pairs_per_image: 每张图片的完整GT pair集合
        k_values: K值列表
        
    Returns:
        包含mean recall统计的字典
    """
    print(f"📊 基于完整GT数据计算谓词级别 Mean Recall (K={k_values})...")
    
    per_image_candidates = get_per_image_candidates(data)
    if per_image_candidates is None:
        return None
    
    # 从GT数据中获取所有谓词类别
    all_predicates = set()
    for gt_pairs in gt_pairs_per_image.values():
        for pair in gt_pairs:
            predicate = pair[2]  # (subject, object, predicate)
            all_predicates.add(predicate)
    predicates = sorted(list(all_predicates))
    
    print(f"   发现 {len(predicates)} 个谓词类别\n")
    
    results = {}
    
    for k in k_values:
        print(f"   计算 Mean Recall@{k}...")
        
        # 初始化每个谓词的统计
        predicate_stats = {pred: {'hit': 0, 'total': 0} for pred in predicates}
        
        for image_id_str, candidates in per_image_candidates.items():
            # 统一image_id类型（转换为整数）
            try:
                image_id = int(image_id_str)
            except (ValueError, TypeError):
                image_id = image_id_str
            
            # 获取该图片的完整GT pair集合
            full_gt_pairs = gt_pairs_per_image.get(image_id, set())
            
            if len(full_gt_pairs) == 0:
                continue
            
            # 获取该图片的物体ID到类别名映射
            image_object_map = object_id_to_class.get(image_id, {})
            
            # 统计该图片中每个谓词类别的GT总数
            gt_predicates_in_image = defaultdict(set)  # predicate -> set of pairs
            for pair in full_gt_pairs:
                predicate = pair[2]  # pair格式: (subject_id, object_id, predicate)
                gt_predicates_in_image[predicate].add(pair)
                predicate_stats[predicate]['total'] += 1
            
            # 过滤掉no relation的预测
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 按相似度排序，取top-k
            sorted_candidates = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            top_k = sorted_candidates[:min(k, len(sorted_candidates))]
            
            # 统计在top-k中被召回的谓词pair（使用物体ID）
            recalled_predicates_in_image = defaultdict(set)  # predicate -> set of recalled pairs
            
            for cand in top_k:
                # 优先使用物体ID
                subject_id = cand.get('subject_id', None)
                object_id = cand.get('object_id', None)
                predicate = cand.get('predicted_predicate', '')
                
                if subject_id is not None and object_id is not None and predicate:
                    pair_key = (subject_id, object_id, predicate)
                    if pair_key in full_gt_pairs:
                        recalled_predicates_in_image[predicate].add(pair_key)
                else:
                    # 向后兼容：如果没有物体ID，尝试使用类别名匹配
                    subject = cand.get('subject', '')
                    object_name = cand.get('object', '')
                    if subject and object_name and predicate:
                        # 尝试找到匹配的物体ID对
                        for gt_subj_id, gt_obj_id, gt_pred in full_gt_pairs:
                            if (gt_pred == predicate and 
                                image_object_map.get(gt_subj_id) == subject and 
                                image_object_map.get(gt_obj_id) == object_name):
                                pair_key = (gt_subj_id, gt_obj_id, predicate)
                                recalled_predicates_in_image[predicate].add(pair_key)
                                break
            
            # 统计每个谓词的召回数（统计所有被召回的pair）
            for predicate, recalled_pairs in recalled_predicates_in_image.items():
                if predicate in predicate_stats:
                    # 统计该谓词在GT中的pair被召回的数量
                    gt_pairs_for_predicate = gt_predicates_in_image.get(predicate, set())
                    recalled_count = len(recalled_pairs & gt_pairs_for_predicate)
                    predicate_stats[predicate]['hit'] += recalled_count
        
        # 计算每个谓词的recall
        per_predicate_recall = {}
        valid_recalls = []
        valid_recalls_base = []  # Base类谓词的recall
        valid_recalls_novel = []  # Novel类谓词的recall
        
        for pred in predicates:
            total = predicate_stats[pred]['total']
            hit = predicate_stats[pred]['hit']
            
            if total > 0:
                recall = hit / total
                category = PREDICATE_CATEGORY_MAPPING.get(pred, "base")  # 默认base
                per_predicate_recall[pred] = {
                    'recall': recall,
                    'hit': hit,
                    'total': total,
                    'category': category
                }
                valid_recalls.append(recall)
                if category == "base":
                    valid_recalls_base.append(recall)
                elif category == "novel":
                    valid_recalls_novel.append(recall)
            else:
                category = PREDICATE_CATEGORY_MAPPING.get(pred, "base")
                per_predicate_recall[pred] = {
                    'recall': 0.0,
                    'hit': 0,
                    'total': 0,
                    'category': category
                }
        
        # 计算mean recall（只对有GT的类别计算）
        mean_recall = np.mean(valid_recalls) if valid_recalls else 0.0
        mean_recall_base = np.mean(valid_recalls_base) if valid_recalls_base else 0.0
        mean_recall_novel = np.mean(valid_recalls_novel) if valid_recalls_novel else 0.0
        
        results[f'mean_recall@{k}'] = {
            'mean_recall': mean_recall,
            'num_valid_predicates': len(valid_recalls),
            'total_predicates': len(predicates),
            'per_predicate_recall': per_predicate_recall,
            # Base和Novel类统计
            'mean_recall_base': mean_recall_base,
            'num_valid_predicates_base': len(valid_recalls_base),
            'mean_recall_novel': mean_recall_novel,
            'num_valid_predicates_novel': len(valid_recalls_novel)
        }
        
        print(f"   Mean Recall@{k:3d}: {mean_recall:.4f} ({mean_recall*100:.2f}%), 有效谓词: {len(valid_recalls)}/{len(predicates)}")
        print(f"   Base类 Mean Recall@{k:3d}: {mean_recall_base:.4f} ({mean_recall_base*100:.2f}%), 有效谓词: {len(valid_recalls_base)}")
        print(f"   Novel类 Mean Recall@{k:3d}: {mean_recall_novel:.4f} ({mean_recall_novel*100:.2f}%), 有效谓词: {len(valid_recalls_novel)}")
    
    print()
    return results


def print_summary(recall_results: Dict, cm_results: Dict, mean_recall_results: Dict = None) -> None:
    """打印总结信息"""
    print("="*80)
    print("📋 总结报告（基于完整GT数据）")
    print("="*80)
    
    print(f"\n召回率统计（基于完整GT数据）:")
    for k in [50, 100]:
        key = f'recall@{k}'
        if key in recall_results:
            result = recall_results[key]
            print(f"  Recall@{k}:")
            print(f"    基于完整GT数据:")
            print(f"      平均召回率: {result['avg_recall']:.4f} ({result['avg_recall']*100:.2f}%)")
            print(f"      整体召回率: {result['overall_recall']:.4f} ({result['overall_recall']*100:.2f}%)")
            print(f"      统计: {result['total_recalled_pairs']}/{result['total_gt_pairs']} pairs")
            print(f"    Base类谓词:")
            print(f"      平均召回率: {result['avg_recall_base']:.4f} ({result['avg_recall_base']*100:.2f}%)")
            print(f"      整体召回率: {result['overall_recall_base']:.4f} ({result['overall_recall_base']*100:.2f}%)")
            print(f"      统计: {result['total_recalled_pairs_base']}/{result['total_gt_pairs_base']} pairs")
            print(f"    Novel类谓词:")
            print(f"      平均召回率: {result['avg_recall_novel']:.4f} ({result['avg_recall_novel']*100:.2f}%)")
            print(f"      整体召回率: {result['overall_recall_novel']:.4f} ({result['overall_recall_novel']*100:.2f}%)")
            print(f"      统计: {result['total_recalled_pairs_novel']}/{result['total_gt_pairs_novel']} pairs")
            print(f"    基于候选列表中的GT pair (对比):")
            print(f"      平均召回率: {result['avg_recall_in_candidates']:.4f} ({result['avg_recall_in_candidates']*100:.2f}%)")
            print(f"      整体召回率: {result['overall_recall_in_candidates']:.4f} ({result['overall_recall_in_candidates']*100:.2f}%)")
            print(f"      统计: {result['total_recalled_pairs_in_candidates']}/{result['total_gt_pairs_in_candidates']} pairs")
            print(f"    Stage1覆盖率: {result['stage1_coverage']:.4f} ({result['stage1_coverage']*100:.2f}%)")
            print(f"    图片数: {result['num_images']}")
    
    print(f"\n混淆矩阵统计 (基于完整GT数据):")
    for k in [50, 100]:
        key = f'confusion_matrix@{k}'
        if key in cm_results:
            result = cm_results[key]
            print(f"  K={k}:")
            print(f"    TP (True Positive): {result['tp']}")
            print(f"    FN (False Negative): {result['fn']}")
            print(f"    FP (False Positive): {result['fp']}")
            print(f"    TN (True Negative): {result['tn']}")
            print(f"    指标 (每张图片平均):")
            print(f"      召回率 (Recall): {result['recall_avg']:.4f} ({result['recall_avg']*100:.2f}%)")
            print(f"      精确率 (Precision): {result['precision_avg']:.4f} ({result['precision_avg']*100:.2f}%)")
            print(f"      F1分数: {result['f1_avg']:.4f}")
            print(f"    指标 (整体累计):")
            print(f"      召回率 (Recall): {result['recall_overall']:.4f} ({result['recall_overall']*100:.2f}%)")
            print(f"      精确率 (Precision): {result['precision_overall']:.4f} ({result['precision_overall']*100:.2f}%)")
            print(f"      F1分数: {result['f1_overall']:.4f}")
            print(f"    完整GT中pair总数: {result['total_gt_pairs']}")
            print(f"    Top-k中GT pair总数: {result['total_top_k_pairs']}")
    
    if mean_recall_results:
        print(f"\nMean Recall统计 (基于完整GT数据):")
        for k in [50, 100]:
            key = f'mean_recall@{k}'
            if key in mean_recall_results:
                result = mean_recall_results[key]
                print(f"  Mean Recall@{k}:")
                print(f"    Mean Recall: {result['mean_recall']:.4f} ({result['mean_recall']*100:.2f}%)")
                print(f"    有效谓词数: {result['num_valid_predicates']}/{result['total_predicates']}")
                print(f"    Base类 Mean Recall: {result['mean_recall_base']:.4f} ({result['mean_recall_base']*100:.2f}%)")
                print(f"    Base类有效谓词数: {result['num_valid_predicates_base']}")
                print(f"    Novel类 Mean Recall: {result['mean_recall_novel']:.4f} ({result['mean_recall_novel']*100:.2f}%)")
                print(f"    Novel类有效谓词数: {result['num_valid_predicates_novel']}")
                
                # 显示Top-10和Bottom-10谓词
                per_predicate = result['per_predicate_recall']
                sorted_predicates = sorted(
                    [(pred, stats) for pred, stats in per_predicate.items() if stats['total'] > 0],
                    key=lambda x: x[1]['recall'],
                    reverse=True
                )
                
                if len(sorted_predicates) > 0:
                    print(f"    Top-10 谓词:")
                    for i, (pred, stats) in enumerate(sorted_predicates[:10], 1):
                        category = stats.get('category', 'unknown')
                        print(f"      {i:2d}. {pred:<20} [{category:5s}] Recall: {stats['recall']:.4f} ({stats['hit']}/{stats['total']})")
                    
                    if len(sorted_predicates) > 10:
                        print(f"    Bottom-10 谓词:")
                        for i, (pred, stats) in enumerate(sorted_predicates[-10:], len(sorted_predicates)-9):
                            category = stats.get('category', 'unknown')
                            print(f"      {i:2d}. {pred:<20} [{category:5s}] Recall: {stats['recall']:.4f} ({stats['hit']}/{stats['total']})")
    
    print()


def export_results(recall_results: Dict, cm_results: Dict, mean_recall_results: Dict = None, output_path: str = None) -> None:
    """导出结果到JSON文件"""
    print(f"💾 正在导出结果到: {output_path}")
    
    export_data = {
        'recall_results': recall_results,
        'confusion_matrix_results': cm_results
    }
    
    if mean_recall_results:
        export_data['mean_recall_results'] = mean_recall_results
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已导出\n")


def main():
    parser = argparse.ArgumentParser(
        description="基于完整GT数据评估场景图关系预测结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本评估（默认计算top50和top100）
  python evaluate_with_gt.py --json_file results.json --gt_file gt.json
  
  # 自定义K值
  python evaluate_with_gt.py --json_file results.json --gt_file gt.json --k-values 50 100 200
  
  # 导出结果
  python evaluate_with_gt.py --json_file results.json --gt_file gt.json --export results.json
        """
    )
    # 
    # 
# INPUT_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_case_20.json"
# OUTPUT_FILE = "/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_20_all.json"

    parser.add_argument('--json_file', type=str, 
                       default='/public/home/wangby2025/plusLab/outputs/test_2000_recall/best_eval_simi_37k_base.json',
                       help='预测结果JSON文件路径')
    parser.add_argument('--gt_file', type=str,
                       default='/public/home/wangby2025/plusLab/VLM2Vec/infer/test_2000_images.json',
                       help='GT文件路径')
    parser.add_argument('--k-values', type=int, nargs='+', default=[50, 100],
                       help='指定要计算的K值列表（默认: 50 100）')
    parser.add_argument('--export', type=str, default=None,
                       help='导出结果到指定JSON文件')
    
    args = parser.parse_args()
    
    # 加载GT数据（返回GT pairs和物体ID到类别名的映射）
    gt_pairs_per_image, object_id_to_class = load_gt_data(args.gt_file)
    
    # 加载结果
    data = load_results(args.json_file)
    
    # 计算召回率
    recall_results = calculate_recall_with_full_gt(data, gt_pairs_per_image, object_id_to_class, args.k_values)
    
    # 计算混淆矩阵
    cm_results = calculate_confusion_matrix_with_full_gt(data, gt_pairs_per_image, object_id_to_class, args.k_values)
    
    # 计算Mean Recall
    mean_recall_results = calculate_mean_recall_with_full_gt(data, gt_pairs_per_image, object_id_to_class, args.k_values)
    
    # 打印总结
    print_summary(recall_results, cm_results, mean_recall_results)
    
    # 导出结果
    if args.export:
        export_results(recall_results, cm_results, mean_recall_results, args.export)
    
    print("="*80)
    print("✅ 评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

