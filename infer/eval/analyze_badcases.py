"""
Badcase分析脚本
用于详细分析推理结果，特别是每个GT pair的相似度分数和排名

主要功能：
1. 对每个GT pair，显示GT谓词在所有50个谓词中的相似度排名
2. 显示GT谓词的相似度分数
3. 分析badcase（GT谓词排名较低的情况）
4. 支持按图片、按pair查看详细信息
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# 50个谓词列表（与推理代码保持一致）
PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind",
    "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
    "in", "in front of", "laying on", "looking at", "lying on", "made of",
    "mounted on", "near", "of", "on", "on back of", "over", "painted on",
    "parked on", "part of", "playing", "riding", "says", "sitting on",
    "standing on", "to", "under", "using", "walking in", "walking on",
    "watching", "wearing", "wears", "with", "no relation"
]


def load_results(json_path: str) -> Dict:
    """加载预测结果JSON文件"""
    print(f"📖 正在加载结果文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 加载完成\n")
    return data


def get_all_candidates_for_pair(image_id, subject_id, object_id, subject, object_name, data: Dict) -> List[Dict]:
    """
    获取某个pair的所有50个谓词候选（按相似度排序）
    
    Args:
        image_id: 图片ID
        subject_id: 主体物体ID
        object_id: 客体物体ID
        subject: 主体类别名
        object_name: 客体类别名
        data: 预测结果数据
    
    Returns:
        该pair的所有候选列表，按相似度降序排列
    """
    # 尝试多种image_id格式
    image_id_str = str(image_id)
    image_id_int = int(image_id) if isinstance(image_id, str) and image_id.isdigit() else image_id
    
    # 优先使用per_image_all_candidates，如果没有则使用per_image_top100_candidates
    candidates = []
    if 'per_image_all_candidates' in data:
        candidates = data['per_image_all_candidates'].get(image_id_str, [])
        if not candidates:
            candidates = data['per_image_all_candidates'].get(image_id_int, [])
    elif 'per_image_top100_candidates' in data:
        candidates = data['per_image_top100_candidates'].get(image_id_str, [])
        if not candidates:
            candidates = data['per_image_top100_candidates'].get(image_id_int, [])
    
    if not candidates:
        return []
    
    # 筛选出该pair的所有候选
    pair_candidates = []
    for cand in candidates:
        # 优先使用物体ID匹配
        cand_subject_id = cand.get('subject_id', None)
        cand_object_id = cand.get('object_id', None)
        
        match = False
        if cand_subject_id is not None and cand_object_id is not None and subject_id is not None and object_id is not None:
            # 使用物体ID匹配
            if cand_subject_id == subject_id and cand_object_id == object_id:
                match = True
        else:
            # 向后兼容：使用类别名匹配
            if cand.get('subject') == subject and cand.get('object') == object_name:
                match = True
        
        if match:
            pair_candidates.append(cand)
    
    # 按相似度排序
    pair_candidates.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    return pair_candidates


def analyze_gt_pair_ranking(image_id, subject_id, object_id, subject, object_name, 
                           gt_predicate: str, data: Dict) -> Dict:
    """
    分析单个GT pair的相似度排名情况
    
    Args:
        image_id: 图片ID
        subject_id: 主体物体ID
        object_id: 客体物体ID
        subject: 主体类别名
        object_name: 客体类别名
        gt_predicate: GT谓词
        data: 预测结果数据
    
    Returns:
        包含排名信息的字典
    """
    # 获取该pair的所有候选
    all_candidates = get_all_candidates_for_pair(
        image_id, subject_id, object_id, subject, object_name, data
    )
    
    if not all_candidates:
        return {
            'error': '未找到该pair的候选数据',
            'gt_predicate': gt_predicate,
            'gt_similarity': None,
            'gt_rank': None,
            'total_predicates': len(PREDICATES)
        }
    
    # 找到GT谓词的相似度和排名
    gt_similarity = None
    gt_rank = None
    
    # 构建谓词到相似度的映射
    predicate_to_similarity = {}
    for cand in all_candidates:
        pred = cand.get('predicted_predicate', '')
        sim = cand.get('similarity', 0)
        if pred not in predicate_to_similarity:
            predicate_to_similarity[pred] = sim
        else:
            # 如果有多个相同谓词，取最大相似度
            predicate_to_similarity[pred] = max(predicate_to_similarity[pred], sim)
    
    # 获取GT谓词的相似度
    if gt_predicate in predicate_to_similarity:
        gt_similarity = predicate_to_similarity[gt_predicate]
    else:
        # GT谓词不在候选列表中
        return {
            'error': f'GT谓词 "{gt_predicate}" 不在候选列表中',
            'gt_predicate': gt_predicate,
            'gt_similarity': None,
            'gt_rank': None,
            'total_predicates': len(PREDICATES),
            'available_predicates': list(predicate_to_similarity.keys())
        }
    
    # 计算排名（按相似度降序）
    all_similarities = sorted(predicate_to_similarity.values(), reverse=True)
    gt_rank = all_similarities.index(gt_similarity) + 1  # 排名从1开始
    
    # 获取Top-10谓词
    top_predicates = sorted(
        predicate_to_similarity.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return {
        'gt_predicate': gt_predicate,
        'gt_similarity': gt_similarity,
        'gt_rank': gt_rank,
        'total_predicates': len(predicate_to_similarity),
        'top_10_predicates': top_predicates,
        'all_predicates_ranked': sorted(
            predicate_to_similarity.items(),
            key=lambda x: x[1],
            reverse=True
        )
    }


def analyze_all_gt_pairs(data: Dict) -> List[Dict]:
    """
    分析所有GT pair的排名情况
    
    Returns:
        每个GT pair的分析结果列表
    """
    print("📊 正在分析所有GT pair的排名情况...")
    
    # 从all_relations中获取所有GT关系
    all_relations = data.get('all_relations', [])
    
    results = []
    for rel in all_relations:
        image_id = rel['image_id']
        subject_id = rel.get('subject_id', None)
        object_id = rel.get('object_id', None)
        subject = rel['subject']
        object_name = rel['object']
        gt_predicate = rel['gt_predicate']
        
        analysis = analyze_gt_pair_ranking(
            image_id, subject_id, object_id, subject, object_name, gt_predicate, data
        )
        
        analysis['image_id'] = image_id
        analysis['subject_id'] = subject_id
        analysis['object_id'] = object_id
        analysis['subject'] = subject
        analysis['object'] = object_name
        
        results.append(analysis)
    
    print(f"✅ 分析了 {len(results)} 个GT pair\n")
    return results


def print_pair_analysis(analysis: Dict, detailed: bool = False):
    """打印单个pair的分析结果"""
    print(f"\n{'='*80}")
    print(f"图片ID: {analysis['image_id']}")
    print(f"配对: {analysis['subject']} (ID:{analysis['subject_id']}) -> {analysis['object']} (ID:{analysis['object_id']})")
    print(f"GT谓词: {analysis['gt_predicate']}")
    print(f"{'='*80}")
    
    if 'error' in analysis:
        print(f"❌ 错误: {analysis['error']}")
        if 'available_predicates' in analysis:
            print(f"   可用谓词: {', '.join(analysis['available_predicates'][:10])}...")
        return
    
    print(f"GT谓词相似度: {analysis['gt_similarity']:.6f}")
    print(f"GT谓词排名: {analysis['gt_rank']}/{analysis['total_predicates']}")
    
    if analysis['gt_rank'] <= 10:
        print(f"✅ GT谓词排名较高（Top-10）")
    elif analysis['gt_rank'] <= 20:
        print(f"⚠️  GT谓词排名中等（Top-20）")
    else:
        print(f"❌ GT谓词排名较低（Top-{analysis['gt_rank']}）")
    
    print(f"\nTop-10 谓词排名:")
    for i, (pred, sim) in enumerate(analysis['top_10_predicates'], 1):
        marker = "✅" if pred == analysis['gt_predicate'] else "  "
        print(f"  {i:2d}. {marker} {pred:20s}: {sim:.6f}")
    
    if detailed:
        print(f"\n所有谓词排名（前20个）:")
        for i, (pred, sim) in enumerate(analysis['all_predicates_ranked'][:20], 1):
            marker = "✅" if pred == analysis['gt_predicate'] else "  "
            print(f"  {i:2d}. {marker} {pred:20s}: {sim:.6f}")


def analyze_badcases(all_analyses: List[Dict], rank_threshold: int = 20) -> Dict:
    """
    分析badcase（GT谓词排名较低的情况）
    
    Args:
        all_analyses: 所有pair的分析结果
        rank_threshold: 排名阈值，超过此值认为是badcase
    
    Returns:
        badcase统计信息
    """
    print(f"\n📊 分析Badcase（排名 > {rank_threshold} 的情况）...")
    
    badcases = []
    good_cases = []
    missing_cases = []
    
    for analysis in all_analyses:
        if 'error' in analysis:
            missing_cases.append(analysis)
        elif analysis['gt_rank'] is None:
            missing_cases.append(analysis)
        elif analysis['gt_rank'] > rank_threshold:
            badcases.append(analysis)
        else:
            good_cases.append(analysis)
    
    # 统计信息
    stats = {
        'total': len(all_analyses),
        'badcases': len(badcases),
        'good_cases': len(good_cases),
        'missing_cases': len(missing_cases),
        'badcase_rate': len(badcases) / len(all_analyses) if all_analyses else 0,
        'badcases_list': badcases
    }
    
    print(f"   总GT pair数: {stats['total']}")
    print(f"   Badcase数（排名>{rank_threshold}）: {stats['badcases']} ({stats['badcase_rate']*100:.2f}%)")
    print(f"   正常case数（排名<={rank_threshold}）: {stats['good_cases']} ({(1-stats['badcase_rate'])*100:.2f}%)")
    print(f"   缺失case数（GT谓词不在候选列表中）: {stats['missing_cases']}")
    
    return stats


def print_badcase_summary(badcases: List[Dict], top_n: int = 20):
    """打印badcase摘要"""
    if not badcases:
        print("\n✅ 没有发现badcase！")
        return
    
    print(f"\n{'='*80}")
    print(f"Badcase摘要（显示前{top_n}个）")
    print(f"{'='*80}")
    
    # 按排名排序
    sorted_badcases = sorted(badcases, key=lambda x: x.get('gt_rank', 999), reverse=True)
    
    for i, analysis in enumerate(sorted_badcases[:top_n], 1):
        print(f"\n{i}. 图片#{analysis['image_id']}: {analysis['subject']} -> {analysis['object']}")
        print(f"   GT谓词: {analysis['gt_predicate']}")
        print(f"   排名: {analysis['gt_rank']}/{analysis['total_predicates']}")
        print(f"   相似度: {analysis['gt_similarity']:.6f}")
        
        # 显示Top-3谓词
        top3 = analysis['top_10_predicates'][:3]
        print(f"   Top-3谓词: {', '.join([f'{p}({s:.4f})' for p, s in top3])}")


def analyze_by_image(data: Dict, image_id: Optional[int] = None):
    """按图片分析"""
    all_analyses = analyze_all_gt_pairs(data)
    
    if image_id is not None:
        # 只分析指定图片
        image_analyses = [a for a in all_analyses if a['image_id'] == image_id]
        print(f"\n📸 图片 #{image_id} 的分析结果（共 {len(image_analyses)} 个GT pair）:")
        print("="*80)
        
        for analysis in image_analyses:
            print_pair_analysis(analysis, detailed=True)
    else:
        # 分析所有图片
        print(f"\n📊 所有图片的分析结果（共 {len(all_analyses)} 个GT pair）:")
        
        # 统计每个图片的badcase数
        image_stats = defaultdict(lambda: {'total': 0, 'badcases': 0, 'good_cases': 0})
        for analysis in all_analyses:
            img_id = analysis['image_id']
            image_stats[img_id]['total'] += 1
            if 'error' not in analysis and analysis.get('gt_rank') is not None:
                if analysis['gt_rank'] > 20:
                    image_stats[img_id]['badcases'] += 1
                else:
                    image_stats[img_id]['good_cases'] += 1
        
        print(f"\n各图片统计（排名>20为badcase）:")
        print(f"{'图片ID':<12} {'总pair数':<10} {'Badcase数':<12} {'正常case数':<12} {'Badcase率':<10}")
        print("-"*60)
        for img_id in sorted(image_stats.keys()):
            stats = image_stats[img_id]
            badcase_rate = stats['badcases'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{img_id:<12} {stats['total']:<10} {stats['badcases']:<12} {stats['good_cases']:<12} {badcase_rate*100:>6.2f}%")


def export_detailed_analysis(all_analyses: List[Dict], output_path: str):
    """导出详细分析结果到JSON文件"""
    print(f"\n💾 正在导出详细分析结果到: {output_path}")
    
    export_data = {
        'summary': {
            'total_pairs': len(all_analyses),
            'badcases_count': sum(1 for a in all_analyses if 'error' not in a and a.get('gt_rank', 999) > 20),
            'good_cases_count': sum(1 for a in all_analyses if 'error' not in a and a.get('gt_rank', 999) <= 20),
            'missing_cases_count': sum(1 for a in all_analyses if 'error' in a)
        },
        'detailed_analyses': all_analyses
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 导出完成")


def main():
    parser = argparse.ArgumentParser(
        description="Badcase分析脚本 - 分析每个GT pair的相似度排名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析所有GT pair的排名情况
  python analyze_badcases.py --json_file results.json
  
  # 分析指定图片
  python analyze_badcases.py --json_file results.json --image_id 2339501
  
  # 分析badcase并导出详细结果
  python analyze_badcases.py --json_file results.json --export badcase_analysis.json
  
  # 设置badcase阈值（默认20）
  python analyze_badcases.py --json_file results.json --rank_threshold 10
        """
    )
    
    parser.add_argument('--json_file', type=str, default="/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_20_all.json",
                       help='预测结果JSON文件路径')
    parser.add_argument('--image_id', type=int, default=None,
                       help='指定要分析的图片ID（可选）')
    parser.add_argument('--rank_threshold', type=int, default=20,
                       help='Badcase排名阈值（默认20，即排名>20认为是badcase）')
    parser.add_argument('--export', type=str, default=None,
                       help='导出详细分析结果到指定JSON文件')
    parser.add_argument('--show_top_badcases', type=int, default=20,
                       help='显示前N个badcase（默认20）')
    
    args = parser.parse_args()
    
    # 加载结果
    data = load_results(args.json_file)
    
    # 分析所有GT pair
    all_analyses = analyze_all_gt_pairs(data)
    
    # 按图片分析或整体分析
    if args.image_id is not None:
        analyze_by_image(data, args.image_id)
    else:
        analyze_by_image(data)
        
        # 分析badcase
        badcase_stats = analyze_badcases(all_analyses, rank_threshold=args.rank_threshold)
        
        # 打印badcase摘要
        print_badcase_summary(badcase_stats['badcases_list'], top_n=args.show_top_badcases)
    
    # 导出结果
    if args.export:
        export_detailed_analysis(all_analyses, args.export)
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()

