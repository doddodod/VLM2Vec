"""
评估场景图关系预测结果的脚本
用于分析 predict_scene_graph_recall.py 输出的 JSON 文件
新任务：统计top50和top100中pair确实存在于GT中的召回率以及混淆矩阵
更新：使用完整的GT数据作为分母，评估真实的召回率
"""

import json
import argparse
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def load_results(json_path: str) -> Dict:
    """加载预测结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_gt_data(gt_file: str) -> Tuple[Dict, Dict]:
    """
    从GT文件中加载所有GT pair以及每张图片的统计信息
    
    Args:
        gt_file: GT文件路径
        
    Returns:
        tuple: (gt_pairs_per_image, image_stats)
        - gt_pairs_per_image: 字典，key为image_id，value为该图片的所有GT pair集合
          GT pair格式: (subject_class, object_class) - 只看subject-object对，不考虑predicate
        - image_stats: 字典，key为image_id，value为{'num_objects': int, 'num_relations': int}
    """
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_pairs_per_image = {}
    image_stats = {}
    
    for item in gt_data:
        image_id = item['image_id']
        objects = {obj['id']: obj['class_name'] for obj in item['objects']}
        relations = item['relations']
        
        # 统计实体数量和关系数量
        num_objects = len(objects)
        num_relations = len(relations)
        image_stats[image_id] = {
            'num_objects': num_objects,
            'num_relations': num_relations
        }
        
        # 构建该图片的所有GT pair（只看subject-object对，不考虑predicate）
        gt_pairs = set()
        for rel in relations:
            subject_id = rel['subject_id']
            object_id = rel['object_id']
            
            if subject_id in objects and object_id in objects:
                subject_class = objects[subject_id]
                object_class = objects[object_id]
                # 只使用 (subject, object) 作为唯一标识，不考虑predicate
                gt_pairs.add((subject_class, object_class))
        
        gt_pairs_per_image[image_id] = gt_pairs
    
    return gt_pairs_per_image, image_stats


def get_per_image_candidates(data: Dict) -> Dict:
    """
    获取按图片分组的候选列表
    
    Returns:
        字典，key为image_id，value为该图片的所有候选列表
    """
    per_image_candidates = {}
    
    if 'per_image_top100_candidates' in data:
        per_image_candidates = data['per_image_top100_candidates']
    elif 'all_candidates' in data:
        all_candidates = data['all_candidates']
        # 按 image_id 分组
        per_image_candidates_list = defaultdict(list)
        for cand in all_candidates:
            per_image_candidates_list[cand['image_id']].append(cand)
        per_image_candidates = dict(per_image_candidates_list)
    else:
        print("⚠️  JSON中没有保存候选列表")
        return None
    
    return per_image_candidates


def calculate_candidate_pair_statistics(data: Dict, gt_pairs_per_image: Dict = None) -> Dict:
    """
    统计所有候选中pair的情况（有GT vs 没有GT）
    
    Args:
        data: 预测结果数据
        gt_pairs_per_image: 每张图片的完整GT pair集合
        
    Returns:
        包含统计信息的字典
    """
    per_image_candidates = get_per_image_candidates(data)
    if per_image_candidates is None:
        return None
    
    # 统计所有候选中的pair
    all_candidate_pairs = set()  # 所有候选中的pair（去重）
    # 记录每个pair在哪些图片中有GT，哪些图片中没有GT
    pair_gt_status = defaultdict(lambda: {'has_gt_in_images': set(), 'no_gt_in_images': set()})
    
    # 按图片统计
    per_image_stats = []
    
    for image_id_str, candidates in per_image_candidates.items():
        # 统一image_id类型（转换为整数）
        try:
            image_id = int(image_id_str)
        except (ValueError, TypeError):
            image_id = image_id_str
        
        # 获取完整的GT pair集合（如果提供）
        full_gt_pairs = None
        if gt_pairs_per_image:
            full_gt_pairs = gt_pairs_per_image.get(image_id, set())
        
        # 统计当前图片的候选pair
        image_candidate_pairs = set()
        image_pairs_with_gt = set()
        image_pairs_without_gt = set()
        
        for cand in candidates:
            # 过滤掉no relation的预测
            if cand.get('predicted_predicate') == 'no relation':
                continue
            
            subject = cand.get('subject', '')
            object_name = cand.get('object', '')
            
            if not subject or not object_name:
                continue
            
            pair_key = (subject, object_name)
            image_candidate_pairs.add(pair_key)
            all_candidate_pairs.add(pair_key)
            
            # 检查是否有GT（针对当前图片）
            has_gt_in_this_image = False
            if full_gt_pairs is not None:
                # 使用完整GT数据判断
                has_gt_in_this_image = pair_key in full_gt_pairs
            else:
                # 如果没有完整GT数据，使用候选中的has_gt标记
                has_gt = cand.get('has_gt', False)
                has_gt_in_this_image = has_gt and cand.get('relation_idx', -1) >= 0
            
            # 记录当前图片的GT状态
            if has_gt_in_this_image:
                image_pairs_with_gt.add(pair_key)
                pair_gt_status[pair_key]['has_gt_in_images'].add(image_id)
            else:
                image_pairs_without_gt.add(pair_key)
                pair_gt_status[pair_key]['no_gt_in_images'].add(image_id)
        
        # 记录每张图片的统计
        per_image_stats.append({
            'image_id': image_id,
            'total_candidate_pairs': len(image_candidate_pairs),
            'pairs_with_gt': len(image_pairs_with_gt),
            'pairs_without_gt': len(image_pairs_without_gt)
        })
    
    # 统计全局：如果pair在至少一张图片中有GT，就算有GT；否则算没有GT
    pairs_with_gt = set()
    pairs_without_gt = set()
    
    for pair_key, status in pair_gt_status.items():
        if len(status['has_gt_in_images']) > 0:
            # 在至少一张图片中有GT，算作有GT
            pairs_with_gt.add(pair_key)
        else:
            # 在所有图片中都没有GT，算作没有GT
            pairs_without_gt.add(pair_key)
    
    total_candidate_pairs = len(all_candidate_pairs)  # 全局去重后的总数
    total_pairs_with_gt = len(pairs_with_gt)
    total_pairs_without_gt = len(pairs_without_gt)
    
    # 计算所有图片候选pair数量的总和（未去重，用于对比）
    total_candidate_pairs_sum = sum(stat['total_candidate_pairs'] for stat in per_image_stats)
    
    # 计算比例
    ratio_with_gt = total_pairs_with_gt / total_candidate_pairs if total_candidate_pairs > 0 else 0.0
    ratio_without_gt = total_pairs_without_gt / total_candidate_pairs if total_candidate_pairs > 0 else 0.0
    
    stats = {
        'total_candidate_pairs': total_candidate_pairs,  # 全局去重后的总数
        'total_candidate_pairs_sum': total_candidate_pairs_sum,  # 所有图片候选pair数量的总和（未去重）
        'pairs_with_gt': total_pairs_with_gt,
        'pairs_without_gt': total_pairs_without_gt,
        'ratio_with_gt': ratio_with_gt,
        'ratio_without_gt': ratio_without_gt,
        'per_image_stats': per_image_stats
    }
    
    return stats


def calculate_recall_at_k(data: Dict, gt_pairs_per_image: Dict = None, image_stats: Dict = None, k_values: List[int] = [50, 100]) -> Dict:
    """
    计算top50和top100中pair确实存在于GT中的召回率
    
    Args:
        data: 预测结果数据
        gt_pairs_per_image: 每张图片的完整GT pair集合（如果提供，使用完整GT作为分母）
        k_values: K值列表，默认[50, 100]
    
    Returns:
        包含召回率统计的字典
    """
    # 静默计算，只在最后输出结果
    
    per_image_candidates = get_per_image_candidates(data)
    if per_image_candidates is None:
        return None
    
    results = {}
    
    for k in k_values:
        
        total_gt_pairs = 0  # 所有GT中的pair总数（完整GT或候选列表中的）
        total_recalled_pairs = 0  # 在top-k中被召回的pair数
        total_gt_pairs_in_candidates = 0  # 在候选列表中的GT pair数（用于对比）
        
        # 用于统计每张图片的情况
        image_recalls = []
        image_recalls_in_candidates = []  # 基于候选列表中的GT pair的召回率（用于对比）
        # 用于统计不同实体数量和关系数量的召回率分布（每个k值重置）
        image_recall_details = []  # 每张图片的详细信息: (image_id, num_objects, num_relations, recall)
        
        for image_id_str, candidates in per_image_candidates.items():
            # 统一image_id类型（转换为整数）
            try:
                image_id = int(image_id_str)
            except (ValueError, TypeError):
                image_id = image_id_str
            
            # 获取完整的GT pair集合（如果提供）
            full_gt_pairs = None
            if gt_pairs_per_image:
                full_gt_pairs = gt_pairs_per_image.get(image_id, set())
                if len(full_gt_pairs) == 0:
                    continue
            
            # 获取候选列表中的GT pair集合（用于对比，只看subject-object对）
            gt_pairs_in_candidates = set()
            for cand in candidates:
                has_gt = cand.get('has_gt', False)
                if has_gt and cand.get('relation_idx', -1) >= 0:
                    subject = cand.get('subject', '')
                    object_name = cand.get('object', '')
                    if subject and object_name:
                        # 只使用 (subject, object) 作为唯一标识，不考虑predicate
                        gt_pairs_in_candidates.add((subject, object_name))
            
            # 如果没有提供完整GT数据，使用候选列表中的GT pair
            if not gt_pairs_per_image:
                if len(gt_pairs_in_candidates) == 0:
                    continue
                full_gt_pairs = gt_pairs_in_candidates
            
            # 过滤掉no relation的预测
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 按相似度排序，取top-k
            sorted_candidates = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            top_k = sorted_candidates[:min(k, len(sorted_candidates))]
            
            # 统计在top-k中确实存在于完整GT中的pair（只看subject-object对）
            recalled_pairs = set()
            for cand in top_k:
                subject = cand.get('subject', '')
                object_name = cand.get('object', '')
                
                if subject and object_name:
                    # 只使用 (subject, object) 作为唯一标识，不考虑predicate
                    pair_key = (subject, object_name)
                    if pair_key in full_gt_pairs:
                        recalled_pairs.add(pair_key)
            
            # 计算基于完整GT的召回率
            recalled_count = len(recalled_pairs)
            gt_count = len(full_gt_pairs)
            recall = recalled_count / gt_count if gt_count > 0 else 0.0
            
            image_recalls.append(recall)
            total_gt_pairs += gt_count
            total_recalled_pairs += recalled_count
            
            # 记录每张图片的详细信息（用于分布统计）
            if image_stats:
                stats = image_stats.get(image_id, {})
                num_objects = stats.get('num_objects', 0)
                num_relations = stats.get('num_relations', 0)
                image_recall_details.append({
                    'image_id': image_id,
                    'num_objects': num_objects,
                    'num_relations': num_relations,
                    'recall': recall,
                    'recalled_pairs': recalled_count,
                    'total_gt_pairs': gt_count
                })
            
            # 计算基于候选列表中GT pair的召回率（用于对比）
            if gt_pairs_per_image and len(gt_pairs_in_candidates) > 0:
                recalled_in_candidates = len(recalled_pairs & gt_pairs_in_candidates)
                gt_in_candidates_count = len(gt_pairs_in_candidates)
                recall_in_candidates = recalled_in_candidates / gt_in_candidates_count if gt_in_candidates_count > 0 else 0.0
                image_recalls_in_candidates.append(recall_in_candidates)
                total_gt_pairs_in_candidates += gt_in_candidates_count
        
        # 计算平均召回率和整体召回率
        avg_recall = np.mean(image_recalls) if image_recalls else 0.0
        overall_recall = total_recalled_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
        
        # 基于候选列表中的GT pair的召回率（用于对比）
        avg_recall_in_candidates = None
        overall_recall_in_candidates = None
        stage1_coverage = None
        if gt_pairs_per_image and total_gt_pairs_in_candidates > 0:
            avg_recall_in_candidates = np.mean(image_recalls_in_candidates) if image_recalls_in_candidates else 0.0
            overall_recall_in_candidates = total_recalled_pairs / total_gt_pairs_in_candidates if total_gt_pairs_in_candidates > 0 else 0.0
            stage1_coverage = total_gt_pairs_in_candidates / total_gt_pairs if total_gt_pairs > 0 else 0.0
        
        results[f'recall@{k}'] = {
            'avg_recall': avg_recall,
            'overall_recall': overall_recall,
            'total_gt_pairs': total_gt_pairs,  # 完整GT pair数
            'total_recalled_pairs': total_recalled_pairs,
            'num_images': len(image_recalls),
            'image_recalls': image_recalls,
            'total_gt_pairs_in_candidates': total_gt_pairs_in_candidates if gt_pairs_per_image else None,
            'avg_recall_in_candidates': avg_recall_in_candidates,
            'overall_recall_in_candidates': overall_recall_in_candidates,
            'stage1_coverage': stage1_coverage,
            'image_recall_details': image_recall_details  # 每张图片的详细信息
        }
        
        # 只在最后总结时输出
    
    print()
    return results


def calculate_confusion_matrix(data: Dict, gt_pairs_per_image: Dict = None, k_values: List[int] = [50, 100]) -> Dict:
    """
    计算pair召回的二分类混淆矩阵
    只关心pair（subject-object对）是否被召回，不区分谓词
    
    Args:
        data: 预测结果数据
        gt_pairs_per_image: 每张图片的完整GT pair集合（如果提供，使用完整GT）
        k_values: K值列表，默认[50, 100]
    
    Returns:
        包含混淆矩阵的字典
    """
    # 混淆矩阵计算（可选，默认不输出）
    
    per_image_candidates = get_per_image_candidates(data)
    if per_image_candidates is None:
        return None
    
    results = {}
    
    for k in k_values:
        
        # 二分类混淆矩阵：pair是否被召回
        # TP: GT中的pair在top-k中出现了
        # FN: GT中的pair在top-k中没有出现
        # FP: top-k中的pair不在GT中
        # TN: top-k中不在GT中的pair
        tp = 0  # True Positive: GT中的pair被召回了
        fn = 0  # False Negative: GT中的pair没有被召回
        fp = 0  # False Positive: top-k中的pair不在GT中
        tn = 0  # True Negative: top-k中不在GT中的pair
        tn_true = 0  # 真正的True Negative: 不在GT中且不在top-k中的pair
        
        # 统计pair出现次数
        gt_pair_occurrences = defaultdict(int)  # GT pair在top-k中出现的次数（全局累计）
        non_gt_pair_occurrences = defaultdict(int)  # 非GT pair在top-k中出现的次数（全局累计）
        gt_pair_occurrences_per_image = []  # 每张图片中GT pair的平均出现次数
        non_gt_pair_occurrences_per_image = []  # 每张图片中非GT pair的平均出现次数
        
        # 每张图片内的指标统计（用于计算平均值）
        image_recalls = []  # 每张图片的召回率
        image_precisions = []  # 每张图片的精确率
        image_f1s = []  # 每张图片的F1分数
        image_accuracies = []  # 每张图片的准确率
        image_specificities = []  # 每张图片的特异性
        
        for image_id_str, candidates in per_image_candidates.items():
            # 统一image_id类型（转换为整数）
            try:
                image_id = int(image_id_str)
            except (ValueError, TypeError):
                image_id = image_id_str
            
            # 获取完整的GT pair集合（如果提供）
            full_gt_pairs = None
            if gt_pairs_per_image:
                full_gt_pairs = gt_pairs_per_image.get(image_id, set())
                if len(full_gt_pairs) == 0:
                    continue
            else:
                # 如果没有提供完整GT数据，从候选列表中统计GT pair（只看subject-object对）
                full_gt_pairs = set()
                for cand in candidates:
                    has_gt = cand.get('has_gt', False)
                    if has_gt and cand.get('relation_idx', -1) >= 0:
                        subject = cand.get('subject', '')
                        object_name = cand.get('object', '')
                        if subject and object_name:
                            # 只使用 (subject, object) 作为唯一标识，不考虑predicate
                            full_gt_pairs.add((subject, object_name))
                
                if len(full_gt_pairs) == 0:
                    continue
            
            # 过滤掉no relation的预测
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 统计所有候选中的pair（去重，用于计算真正的TN）
            all_candidate_pairs = set()  # 所有候选中的pair（使用subject-object对作为标识）
            for cand in non_bg_candidates:
                subject = cand.get('subject', '')
                object_name = cand.get('object', '')
                pair_key = (subject, object_name)
                all_candidate_pairs.add(pair_key)
            
            # 按相似度排序，取top-k
            sorted_candidates = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            top_k = sorted_candidates[:min(k, len(sorted_candidates))]
            
            # 统计top-k中出现的pair（只看subject-object对）
            top_k_pairs = set()  # top-k中在完整GT中的pair
            top_k_non_gt_pairs = set()  # top-k中不在完整GT中的pair
            top_k_pair_keys = set()  # top-k中所有pair的key (subject, object)
            
            # 统计当前图片中pair的出现次数
            image_gt_pair_counts = defaultdict(int)  # 当前图片中GT pair的出现次数
            image_non_gt_pair_counts = defaultdict(int)  # 当前图片中非GT pair的出现次数
            
            for cand in top_k:
                subject = cand.get('subject', '')
                object_name = cand.get('object', '')
                pair_key_so = (subject, object_name)  # subject-object对（只看这个，不考虑predicate）
                
                if not subject or not object_name:
                    continue
                
                top_k_pair_keys.add(pair_key_so)
                
                if pair_key_so in full_gt_pairs:
                    # 在完整GT中的pair
                    top_k_pairs.add(pair_key_so)
                    image_gt_pair_counts[pair_key_so] += 1
                    # 全局统计（用于跨图片统计）
                    gt_pair_occurrences[pair_key_so] += 1
                else:
                    # 不在完整GT中的pair
                    top_k_non_gt_pairs.add(pair_key_so)
                    image_non_gt_pair_counts[pair_key_so] += 1
                    # 全局统计（用于跨图片统计）
                    non_gt_pair_occurrences[pair_key_so] += 1
            
            # 计算当前图片的平均出现次数
            if len(image_gt_pair_counts) > 0:
                image_avg_gt_occurrences = sum(image_gt_pair_counts.values()) / len(image_gt_pair_counts)
                # 累加到全局统计
                gt_pair_occurrences_per_image.append(image_avg_gt_occurrences)
            
            if len(image_non_gt_pair_counts) > 0:
                image_avg_non_gt_occurrences = sum(image_non_gt_pair_counts.values()) / len(image_non_gt_pair_counts)
                # 累加到全局统计
                non_gt_pair_occurrences_per_image.append(image_avg_non_gt_occurrences)
            
            # 计算当前图片的TP、FN、FP、TN（基于完整GT）
            # TP: 完整GT中的pair在top-k中出现了
            image_tp = len(top_k_pairs)
            # FN: 完整GT中的pair在top-k中没有出现
            image_fn = len(full_gt_pairs - top_k_pairs)
            # FP: top-k中的pair不在完整GT中
            image_fp = len(top_k_non_gt_pairs)
            # TN: top-k中不在完整GT中的pair（这里统计的是subject-object对）
            image_tn = len(top_k_non_gt_pairs)
            
            # 累计到全局统计
            tp += image_tp
            fn += image_fn
            fp += image_fp
            tn += image_tn
            
            # 计算当前图片的指标
            image_total_gt = image_tp + image_fn
            image_total_topk_gt = image_tp + image_fp
            
            image_recall = 0.0
            image_precision = 0.0
            
            if image_total_gt > 0:
                image_recall = image_tp / image_total_gt
                image_recalls.append(image_recall)
            
            if image_total_topk_gt > 0:
                image_precision = image_tp / image_total_topk_gt
                image_precisions.append(image_precision)
            
            # 计算F1分数
            if image_precision > 0 and image_recall > 0:
                image_f1 = 2 * image_precision * image_recall / (image_precision + image_recall)
                image_f1s.append(image_f1)
            
            image_total = image_tp + image_tn + image_fp + image_fn
            if image_total > 0:
                image_accuracy = (image_tp + image_tn) / image_total
                image_accuracies.append(image_accuracy)
            
            if (image_tn + image_fp) > 0:
                image_specificity = image_tn / (image_tn + image_fp)
                image_specificities.append(image_specificity)
            
            # 计算真正的TN：不在GT中且不在top-k中的pair
            # 所有候选pair中不在完整GT中的pair
            # 注意：这里我们只统计subject-object对，不考虑谓词
            all_non_gt_pairs = all_candidate_pairs
            # 不在完整GT中且不在top-k中的pair
            true_tn_pairs = all_non_gt_pairs - top_k_pair_keys
            tn_true += len(true_tn_pairs)
        
        # 构建2x2混淆矩阵
        #           预测为正例(在top-k中)  预测为负例(不在top-k中)
        # 实际为正例(在GT中)    TP              FN
        # 实际为负例(不在GT中)  FP              TN
        # 注意：TN统计的是top-k中不在GT中的pair（relation_idx == -1）
        # 真正的TN应该是"不在GT中且不在top-k中"，但我们无法统计所有可能的pair
        cm = np.array([[tp, fn],
                       [fp, tn]])
        
        # 计算指标
        total_gt_pairs = tp + fn  # 完整GT中的pair总数
        total_top_k_pairs = tp  # top-k中在完整GT中的pair总数
        total_top_k_non_gt_pairs = fp  # top-k中不在完整GT中的pair总数
        total_top_k_pairs_all = tp + fp  # top-k中的总pair数（TP + FP）
        
        # 计算pair平均出现次数
        # 方式1：跨所有图片的全局平均（每个唯一pair在所有图片中的平均出现次数）
        total_gt_pair_occurrences = sum(gt_pair_occurrences.values())  # GT pair在top-k中的总出现次数（跨所有图片）
        total_non_gt_pair_occurrences = sum(non_gt_pair_occurrences.values())  # 非GT pair在top-k中的总出现次数（跨所有图片）
        unique_gt_pairs_in_topk = len(gt_pair_occurrences)  # top-k中唯一的GT pair数量（跨所有图片）
        unique_non_gt_pairs_in_topk = len(non_gt_pair_occurrences)  # top-k中唯一的非GT pair数量（跨所有图片）
        
        avg_gt_pair_occurrences_global = total_gt_pair_occurrences / unique_gt_pairs_in_topk if unique_gt_pairs_in_topk > 0 else 0.0
        avg_non_gt_pair_occurrences_global = total_non_gt_pair_occurrences / unique_non_gt_pairs_in_topk if unique_non_gt_pairs_in_topk > 0 else 0.0
        
        # 方式2：每张图片内的平均（在每张图片的top-k中，每个pair平均出现多少次）
        avg_gt_pair_occurrences_per_image = np.mean(gt_pair_occurrences_per_image) if gt_pair_occurrences_per_image else 0.0
        avg_non_gt_pair_occurrences_per_image = np.mean(non_gt_pair_occurrences_per_image) if non_gt_pair_occurrences_per_image else 0.0
        
        # 整体指标（跨所有图片累计）
        # 召回率 (Recall/Sensitivity): TP / (TP + FN)
        recall_overall = tp / total_gt_pairs if total_gt_pairs > 0 else 0.0
        
        # 精确率 (Precision): TP / (TP + FP)
        precision_overall = tp / total_top_k_pairs if total_top_k_pairs > 0 else 0.0
        
        # F1分数
        f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall) if (precision_overall + recall_overall) > 0 else 0.0
        
        # 准确率 (Accuracy): (TP + TN) / (TP + TN + FP + FN)
        accuracy_overall = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # 特异性 (Specificity): TN / (TN + FP)
        specificity_overall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 平均指标（每张图片的平均值）
        recall_avg = np.mean(image_recalls) if image_recalls else 0.0
        precision_avg = np.mean(image_precisions) if image_precisions else 0.0
        f1_avg = np.mean(image_f1s) if image_f1s else 0.0
        accuracy_avg = np.mean(image_accuracies) if image_accuracies else 0.0
        specificity_avg = np.mean(image_specificities) if image_specificities else 0.0
        
        results[f'confusion_matrix@{k}'] = {
            'matrix': cm.tolist(),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'tn_true': int(tn_true),  # 真正的TN: 不在GT中且不在top-k中的pair
            'total_gt_pairs': int(total_gt_pairs),
            'total_top_k_pairs': int(total_top_k_pairs),
            'total_top_k_non_gt_pairs': int(total_top_k_non_gt_pairs),
            'total_top_k_pairs_all': int(total_top_k_pairs_all),
            'avg_gt_pair_occurrences_per_image': avg_gt_pair_occurrences_per_image,  # GT pair在每张图片top-k中的平均出现次数
            'avg_non_gt_pair_occurrences_per_image': avg_non_gt_pair_occurrences_per_image,  # 非GT pair在每张图片top-k中的平均出现次数
            'avg_gt_pair_occurrences_global': avg_gt_pair_occurrences_global,  # GT pair跨所有图片的平均出现次数
            'avg_non_gt_pair_occurrences_global': avg_non_gt_pair_occurrences_global,  # 非GT pair跨所有图片的平均出现次数
            'total_gt_pair_occurrences': int(total_gt_pair_occurrences),  # GT pair在top-k中的总出现次数
            'total_non_gt_pair_occurrences': int(total_non_gt_pair_occurrences),  # 非GT pair在top-k中的总出现次数
            'unique_gt_pairs_in_topk': int(unique_gt_pairs_in_topk),  # top-k中唯一的GT pair数量
            'unique_non_gt_pairs_in_topk': int(unique_non_gt_pairs_in_topk),  # top-k中唯一的非GT pair数量
            # 整体指标（跨所有图片累计）
            'recall_overall': recall_overall,
            'precision_overall': precision_overall,
            'f1_overall': f1_overall,
            'accuracy_overall': accuracy_overall,
            'specificity_overall': specificity_overall,
            # 平均指标（每张图片的平均值）
            'recall_avg': recall_avg,
            'precision_avg': precision_avg,
            'f1_avg': f1_avg,
            'accuracy_avg': accuracy_avg,
            'specificity_avg': specificity_avg,
            # 向后兼容（使用整体指标）
            'recall': recall_overall,
            'precision': precision_overall,
            'f1': f1_overall,
            'accuracy': accuracy_overall,
            'specificity': specificity_overall
        }
        
        print(f"   混淆矩阵 (2x2):")
        print(f"               预测为正例(在top-k)  预测为负例(不在top-k)")
        print(f"   实际为正例(在GT)    TP={tp:6d}        FN={fn:6d}")
        print(f"   实际为负例(不在GT)  FP={fp:6d}        TN={tn:6d}")
        print(f"   指标 (每张图片平均):")
        print(f"     召回率 (Recall): {recall_avg:.4f} ({recall_avg*100:.2f}%)")
        print(f"     精确率 (Precision): {precision_avg:.4f} ({precision_avg*100:.2f}%)")
        print(f"     特异性 (Specificity): {specificity_avg:.4f} ({specificity_avg*100:.2f}%)")
        print(f"     F1分数: {f1_avg:.4f}")
        print(f"     准确率 (Accuracy): {accuracy_avg:.4f} ({accuracy_avg*100:.2f}%)")
        print(f"   指标 (整体累计):")
        print(f"     召回率 (Recall): {recall_overall:.4f} ({recall_overall*100:.2f}%)")
        print(f"     精确率 (Precision): {precision_overall:.4f} ({precision_overall*100:.2f}%)")
        print(f"     特异性 (Specificity): {specificity_overall:.4f} ({specificity_overall*100:.2f}%)")
        print(f"     F1分数: {f1_overall:.4f}")
        print(f"     准确率 (Accuracy): {accuracy_overall:.4f} ({accuracy_overall*100:.2f}%)")
        print(f"   统计: GT中有{total_gt_pairs}个pair, top-k中有{total_top_k_pairs_all}个pair (其中{total_top_k_pairs}个GT pair, {total_top_k_non_gt_pairs}个非GT pair)")
        print(f"   真正的TN (不在GT中且不在top-k中): {tn_true}")
        print(f"   Pair出现次数统计 (每张图片内):")
        print(f"     GT pair平均出现次数: {avg_gt_pair_occurrences_per_image:.4f} (在每张图片的top-{k}中)")
        print(f"     非GT pair平均出现次数: {avg_non_gt_pair_occurrences_per_image:.4f} (在每张图片的top-{k}中)")
        print(f"   Pair出现次数统计 (跨所有图片):")
        print(f"     GT pair平均出现次数: {avg_gt_pair_occurrences_global:.4f} (总出现{total_gt_pair_occurrences}次, 唯一{unique_gt_pairs_in_topk}个pair)")
        print(f"     非GT pair平均出现次数: {avg_non_gt_pair_occurrences_global:.4f} (总出现{total_non_gt_pair_occurrences}次, 唯一{unique_non_gt_pairs_in_topk}个pair)")
    
    print()
    return results


def display_confusion_matrix(cm_data: Dict, k: int, output_path: str = None):
    """
    显示2x2混淆矩阵（pair召回）
    
    Args:
        cm_data: 混淆矩阵数据
        k: K值
        output_path: 保存图片的路径（可选）
    """
    if not HAS_PLOTTING:
        print(f"   ⚠️  未安装matplotlib/seaborn，无法显示混淆矩阵图片")
        return
    
    key = f'confusion_matrix@{k}'
    if key not in cm_data:
        return
    
    cm = np.array(cm_data[key]['matrix'])
    
    # 归一化混淆矩阵（按行归一化，显示召回率）
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm_normalized, 
                annot=cm,  # 显示原始数值
                fmt='d',
                cmap='Blues',
                xticklabels=['预测为正例(在top-k)', '预测为负例(不在top-k)'],
                yticklabels=['实际为正例(在GT)', '实际为负例(不在GT)'],
                cbar_kws={'label': '归一化值'})
    plt.title(f'Pair召回混淆矩阵 (Top-{k})', fontsize=14, fontweight='bold')
    plt.xlabel('预测', fontsize=12)
    plt.ylabel('实际', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   混淆矩阵已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_candidate_pair_statistics(candidate_stats: Dict) -> None:
    """打印候选pair统计信息"""
    if candidate_stats is None:
        print("⚠️  无法统计候选pair信息")
        return
    
    print(f"\n{'='*80}")
    print("所有候选中的Pair统计（只看subject-object对，不考虑predicate）")
    print(f"{'='*80}")
    print(f"总候选pair数（全局去重）: {candidate_stats['total_candidate_pairs']:,}")
    if 'total_candidate_pairs_sum' in candidate_stats:
        print(f"总候选pair数（所有图片总和，未去重）: {candidate_stats['total_candidate_pairs_sum']:,}")
        if candidate_stats['total_candidate_pairs_sum'] > 0:
            dedup_ratio = candidate_stats['total_candidate_pairs'] / candidate_stats['total_candidate_pairs_sum']
            print(f"  去重比例: {dedup_ratio:.2%} (说明有 {candidate_stats['total_candidate_pairs_sum'] - candidate_stats['total_candidate_pairs']:,} 个pair在多张图片中重复出现)")
    print(f"有GT的pair数: {candidate_stats['pairs_with_gt']:,} ({candidate_stats['ratio_with_gt']:.2%})")
    print(f"没有GT的pair数: {candidate_stats['pairs_without_gt']:,} ({candidate_stats['ratio_without_gt']:.2%})")
    print(f"{'='*80}\n")


def print_summary(recall_results: Dict, cm_results: Dict = None) -> None:
    """打印总结信息"""
    # 显示所有K值的召回率
    k_values = sorted([int(k.split('@')[1]) for k in recall_results.keys() if k.startswith('recall@')])
    for k in k_values:
        key = f'recall@{k}'
        if key in recall_results:
            result = recall_results[key]
            print(f"Recall@{k}: {result['overall_recall']:.4f} ({result['overall_recall']*100:.2f}%) - {result['total_recalled_pairs']}/{result['total_gt_pairs']} pairs")


def print_recall_distribution_by_image_stats(recall_results: Dict, k: int = None) -> None:
    """
    打印按实体数量和关系数量分组的召回率分布表
    
    Args:
        recall_results: 召回率结果字典
        k: 指定的K值，如果为None则显示所有K值
    """
    k_values = sorted([int(k.split('@')[1]) for k in recall_results.keys() if k.startswith('recall@')])
    
    if k is not None:
        k_values = [k] if k in k_values else []
    
    for k_val in k_values:
        key = f'recall@{k_val}'
        if key not in recall_results:
            continue
        
        result = recall_results[key]
        image_recall_details = result.get('image_recall_details', [])
        
        if not image_recall_details:
            print(f"\n⚠️  Top-{k_val}: 没有图片详细信息，无法生成分布表")
            continue
        
        # 按实体数量和关系数量分组
        # 定义分组区间
        object_bins = [
            (0, 5, "1-5"),
            (6, 10, "6-10"),
            (11, 15, "11-15"),
            (16, 20, "16-20"),
            (21, 30, "21-30"),
            (31, 50, "31-50"),
            (51, 100, "51-100"),
            (101, float('inf'), "101+")
        ]
        
        relation_bins = [
            (0, 5, "1-5"),
            (6, 10, "6-10"),
            (11, 15, "11-15"),
            (16, 20, "16-20"),
            (21, 30, "21-30"),
            (31, 50, "31-50"),
            (51, 100, "51-100"),
            (101, float('inf'), "101+")
        ]
        
        def get_bin_label(value, bins):
            """根据值返回对应的分组标签"""
            # 处理0值的情况
            if value == 0:
                return "0"
            for min_val, max_val, label in bins:
                if min_val <= value <= max_val:
                    return label
            return "未知"
        
        # 统计每个分组的召回率
        group_stats = defaultdict(lambda: {
            'recalls': [],
            'total_recalled_pairs': 0,
            'total_gt_pairs': 0,
            'num_images': 0
        })
        
        for detail in image_recall_details:
            num_objects = detail['num_objects']
            num_relations = detail['num_relations']
            recall = detail['recall']
            recalled_pairs = detail['recalled_pairs']
            total_gt_pairs = detail['total_gt_pairs']
            
            obj_bin = get_bin_label(num_objects, object_bins)
            rel_bin = get_bin_label(num_relations, relation_bins)
            
            group_key = (obj_bin, rel_bin)
            group_stats[group_key]['recalls'].append(recall)
            group_stats[group_key]['total_recalled_pairs'] += recalled_pairs
            group_stats[group_key]['total_gt_pairs'] += total_gt_pairs
            group_stats[group_key]['num_images'] += 1
        
        # 打印分布表
        print(f"\n{'='*80}")
        print(f"Top-{k_val}: 按实体数量和关系数量分组的GT Pair召回率分布表")
        print(f"{'='*80}")
        
        def sort_bin_key(x):
            """用于排序分组标签的函数"""
            if x == "0":
                return 0
            if '-' in x:
                return int(x.split('-')[0])
            if '+' in x:
                return int(x.replace('+', ''))
            if x == "未知":
                return 9999
            return 999
        
        # 获取所有唯一的关系数量分组和实体数量分组
        unique_obj_bins = sorted(set([obj_bin for obj_bin, _ in group_stats.keys()]), key=sort_bin_key)
        unique_rel_bins = sorted(set([rel_bin for _, rel_bin in group_stats.keys()]), key=sort_bin_key)
        
        # 打印表头
        header = f"{'关系数量':<12}"
        for obj_bin in unique_obj_bins:
            header += f" | {obj_bin:>12}"
        header += f" | {'平均':>12}"
        print(header)
        print("-" * len(header))
        
        # 打印每一行（按关系数量分组）
        for rel_bin in unique_rel_bins:
            row = f"{rel_bin:<12}"
            row_recalls = []
            for obj_bin in unique_obj_bins:
                group_key = (obj_bin, rel_bin)
                if group_key in group_stats:
                    stats = group_stats[group_key]
                    avg_recall = np.mean(stats['recalls']) if stats['recalls'] else 0.0
                    overall_recall = stats['total_recalled_pairs'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
                    # 使用整体召回率（更准确）
                    recall_to_show = overall_recall
                    row_recalls.append(recall_to_show)
                    row += f" | {recall_to_show:>11.2%}"
                else:
                    row += f" | {'-':>12}"
            
            # 计算该行的平均召回率
            if row_recalls:
                row_avg = np.mean(row_recalls)
                row += f" | {row_avg:>11.2%}"
            else:
                row += f" | {'-':>12}"
            print(row)
        
        # 打印列平均
        print("-" * len(header))
        col_avg_row = f"{'平均':<12}"
        for obj_bin in unique_obj_bins:
            col_recalls = []
            for rel_bin in unique_rel_bins:
                group_key = (obj_bin, rel_bin)
                if group_key in group_stats:
                    stats = group_stats[group_key]
                    overall_recall = stats['total_recalled_pairs'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
                    col_recalls.append(overall_recall)
            if col_recalls:
                col_avg = np.mean(col_recalls)
                col_avg_row += f" | {col_avg:>11.2%}"
            else:
                col_avg_row += f" | {'-':>12}"
        
        # 总体平均
        all_recalls = []
        for stats in group_stats.values():
            if stats['total_gt_pairs'] > 0:
                overall_recall = stats['total_recalled_pairs'] / stats['total_gt_pairs']
                all_recalls.append(overall_recall)
        if all_recalls:
            overall_avg = np.mean(all_recalls)
            col_avg_row += f" | {overall_avg:>11.2%}"
        else:
            col_avg_row += f" | {'-':>12}"
        print(col_avg_row)
        
        # 打印每个分组的详细信息（图片数量、GT pair数量等）
        print(f"\n详细统计信息:")
        print(f"{'实体数量':<12} | {'关系数量':<12} | {'图片数':>8} | {'GT Pairs':>12} | {'召回Pairs':>12} | {'召回率':>10}")
        print("-" * 80)
        def sort_group_key(x):
            """用于排序分组键的函数"""
            obj_bin, rel_bin = x[0]
            obj_key = sort_bin_key(obj_bin)
            rel_key = sort_bin_key(rel_bin)
            return (obj_key, rel_key)
        
        for (obj_bin, rel_bin), stats in sorted(group_stats.items(), key=sort_group_key):
            overall_recall = stats['total_recalled_pairs'] / stats['total_gt_pairs'] if stats['total_gt_pairs'] > 0 else 0.0
            print(f"{obj_bin:<12} | {rel_bin:<12} | {stats['num_images']:>8} | {stats['total_gt_pairs']:>12} | {stats['total_recalled_pairs']:>12} | {overall_recall:>9.2%}")
        
        print(f"{'='*80}\n")


def export_results(recall_results: Dict, cm_results: Dict = None, output_path: str = None) -> None:
    """导出结果到JSON文件"""
    print(f"💾 正在导出结果到: {output_path}")
    
    export_data = {
        'recall_results': recall_results,
        'confusion_matrix_results': cm_results
    }
    
    # 将numpy数组转换为列表
    if cm_results:
        for k in [50, 100]:
            key = f'confusion_matrix@{k}'
            if key in cm_results:
                # matrix已经是list了，不需要转换
                pass
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已导出\n")


def main():
    parser = argparse.ArgumentParser(
        description="评估场景图关系预测结果 - Top-K召回率和混淆矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本评估（默认计算top50和top100）
  python topk_relation_filter.py --json_file results.json
  
  # 自定义K值
  python topk_relation_filter.py --json_file results.json --k-values 50 100 200
  
  # 保存混淆矩阵图片
  python topk_relation_filter.py --json_file results.json --save-cm-fig
  
  # 导出结果
  python topk_relation_filter.py --json_file results.json --export results.json
        """
    )
    
    parser.add_argument('--json_file', type=str, 
                       default='/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_2000.json',
                       help='预测结果JSON文件路径')
    parser.add_argument('--gt_file', type=str, default='/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_2000_images.json',
                       help='GT文件路径（如果提供，将使用完整GT数据作为分母）')
    parser.add_argument('--k-values', type=int, nargs='+', default=[50, 100,20000],
                       help='指定要计算的K值列表（默认: 50 100）')
    parser.add_argument('--export', type=str, default=None,
                       help='导出结果到指定JSON文件')
    parser.add_argument('--save-cm-fig', action='store_true',
                       help='保存混淆矩阵图片')
    parser.add_argument('--cm-fig-dir', type=str, default='./',
                       help='混淆矩阵图片保存目录（默认: ./）')
    
    args = parser.parse_args()
    
    # 加载GT数据（如果提供）
    gt_pairs_per_image = None
    image_stats = None
    if args.gt_file:
        gt_pairs_per_image, image_stats = load_gt_data(args.gt_file)
    
    # 加载结果
    data = load_results(args.json_file)
    
    # 统计所有候选中的pair情况（有GT vs 没有GT）
    candidate_stats = calculate_candidate_pair_statistics(data, gt_pairs_per_image)
    print_candidate_pair_statistics(candidate_stats)
    
    # 计算召回率
    recall_results = calculate_recall_at_k(data, gt_pairs_per_image, image_stats, args.k_values)
    
    # 打印总结（只显示召回率）
    print_summary(recall_results)
    
    # 打印按实体数量和关系数量分组的召回率分布表
    print_recall_distribution_by_image_stats(recall_results)
    
    # 可选：计算混淆矩阵（如果需要）
    cm_results = None
    if args.save_cm_fig:
        cm_results = calculate_confusion_matrix(data, gt_pairs_per_image, args.k_values)
        import os
        os.makedirs(args.cm_fig_dir, exist_ok=True)
        for k in args.k_values:
            fig_path = os.path.join(args.cm_fig_dir, f'confusion_matrix_top{k}.png')
            display_confusion_matrix(cm_results, k, output_path=fig_path)
    
    # 导出结果
    if args.export:
        export_results(recall_results, cm_results, args.export)
    
    print("✅ 评估完成")


if __name__ == "__main__":
    main()
