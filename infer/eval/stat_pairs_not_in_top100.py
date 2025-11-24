"""
统计一个pair的任一谓词一次都没有进入过top100的pair数量

对于每个配对(subject, object)，检查该配对的所有50个谓词预测中，
是否有任何一个进入过top100。如果没有，则统计这个pair。
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, Set, Tuple


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
    
    # 优先使用所有候选（如果存在）
    if 'per_image_all_candidates' in data:
        print("   使用 per_image_all_candidates 字段（所有候选）...")
        per_image_candidates = data['per_image_all_candidates']
    elif 'per_image_top100_candidates' in data:
        print("   使用 per_image_top100_candidates 字段（Top-100候选）...")
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
        print("⚠️  JSON中没有保存候选列表")
        return None
    
    return per_image_candidates


def get_top100_candidates(data: Dict) -> Dict:
    """
    获取每张图片的Top-100候选
    
    Returns:
        字典，key为image_id，value为该图片的Top-100候选列表
    """
    if 'per_image_top100_candidates' in data:
        return data['per_image_top100_candidates']
    elif 'per_image_all_candidates' in data:
        # 从所有候选中选择Top-100
        per_image_top100 = {}
        for image_id, candidates in data['per_image_all_candidates'].items():
            sorted_candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
            per_image_top100[image_id] = sorted_candidates[:min(100, len(sorted_candidates))]
        return per_image_top100
    else:
        return None


def count_pairs_not_in_top100(data: Dict) -> Dict:
    """
    统计一个pair的任一谓词一次都没有进入过top100的pair数量
    
    Args:
        data: 预测结果数据
        
    Returns:
        统计结果字典
    """
    print("📊 统计从未进入Top-100的pair数量...\n")
    
    # 获取所有候选和Top-100候选
    per_image_all_candidates = get_per_image_candidates(data)
    per_image_top100_candidates = get_top100_candidates(data)
    
    if per_image_all_candidates is None:
        print("❌ 无法获取候选列表")
        return None
    
    if per_image_top100_candidates is None:
        print("❌ 无法获取Top-100候选列表")
        return None
    
    # 统计结果
    total_pairs = 0  # 总pair数
    pairs_in_top100 = 0  # 至少有一个谓词进入过Top-100的pair数
    pairs_not_in_top100 = 0  # 所有谓词都没有进入过Top-100的pair数
    
    # 按图片统计
    per_image_stats = []
    
    for image_id in per_image_all_candidates.keys():
        all_candidates = per_image_all_candidates.get(image_id, [])
        top100_candidates = per_image_top100_candidates.get(image_id, [])
        
        # 构建Top-100中的pair集合（用于快速查找）
        top100_pairs = set()
        for cand in top100_candidates:
            subject = cand.get('subject', '')
            object_name = cand.get('object', '')
            if subject and object_name:
                top100_pairs.add((subject, object_name))
        
        # 统计该图片的所有pair
        image_pairs = set()
        for cand in all_candidates:
            subject = cand.get('subject', '')
            object_name = cand.get('object', '')
            if subject and object_name:
                image_pairs.add((subject, object_name))
        
        # 统计该图片中从未进入Top-100的pair
        image_pairs_not_in_top100 = image_pairs - top100_pairs
        
        total_pairs += len(image_pairs)
        pairs_in_top100 += len(top100_pairs)
        pairs_not_in_top100 += len(image_pairs_not_in_top100)
        
        per_image_stats.append({
            'image_id': image_id,
            'total_pairs': len(image_pairs),
            'pairs_in_top100': len(top100_pairs),
            'pairs_not_in_top100': len(image_pairs_not_in_top100),
            'pairs_not_in_top100_list': list(image_pairs_not_in_top100)
        })
    
    # 计算比例
    pairs_in_top100_rate = pairs_in_top100 / total_pairs if total_pairs > 0 else 0.0
    pairs_not_in_top100_rate = pairs_not_in_top100 / total_pairs if total_pairs > 0 else 0.0
    
    results = {
        'total_pairs': total_pairs,
        'pairs_in_top100': pairs_in_top100,
        'pairs_not_in_top100': pairs_not_in_top100,
        'pairs_in_top100_rate': pairs_in_top100_rate,
        'pairs_not_in_top100_rate': pairs_not_in_top100_rate,
        'per_image_stats': per_image_stats
    }
    
    return results


def print_statistics(results: Dict):
    """打印统计结果"""
    print("="*80)
    print("📊 Pair统计结果：从未进入Top-100的Pair数量")
    print("="*80)
    print(f"\n总体统计:")
    print(f"  总pair数: {results['total_pairs']}")
    print(f"  至少有一个谓词进入过Top-100的pair数: {results['pairs_in_top100']} ({results['pairs_in_top100_rate']*100:.2f}%)")
    print(f"  所有谓词都没有进入过Top-100的pair数: {results['pairs_not_in_top100']} ({results['pairs_not_in_top100_rate']*100:.2f}%)")
    
    # 统计每张图片的情况
    per_image_stats = results['per_image_stats']
    if per_image_stats:
        print(f"\n每张图片统计（前10张）:")
        print(f"{'图片ID':<15}{'总pair数':<12}{'Top-100中':<12}{'不在Top-100':<15}")
        print("-"*60)
        for stat in per_image_stats[:10]:
            print(f"{str(stat['image_id']):<15}{stat['total_pairs']:<12}{stat['pairs_in_top100']:<12}{stat['pairs_not_in_top100']:<15}")
        
        if len(per_image_stats) > 10:
            print(f"\n... (共 {len(per_image_stats)} 张图片)")
        
        # 统计有pair不在Top-100的图片数量
        images_with_pairs_not_in_top100 = sum(1 for stat in per_image_stats if stat['pairs_not_in_top100'] > 0)
        print(f"\n有pair不在Top-100的图片数: {images_with_pairs_not_in_top100}/{len(per_image_stats)}")
        
        # 统计平均每张图片的情况
        avg_total_pairs = sum(stat['total_pairs'] for stat in per_image_stats) / len(per_image_stats)
        avg_pairs_in_top100 = sum(stat['pairs_in_top100'] for stat in per_image_stats) / len(per_image_stats)
        avg_pairs_not_in_top100 = sum(stat['pairs_not_in_top100'] for stat in per_image_stats) / len(per_image_stats)
        
        print(f"\n平均每张图片:")
        print(f"  平均总pair数: {avg_total_pairs:.2f}")
        print(f"  平均Top-100中的pair数: {avg_pairs_in_top100:.2f}")
        print(f"  平均不在Top-100的pair数: {avg_pairs_not_in_top100:.2f}")
    
    print("="*80)


def export_results(results: Dict, output_path: str):
    """导出结果到JSON文件"""
    print(f"\n💾 正在导出结果到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已导出\n")


def main():
    parser = argparse.ArgumentParser(
        description="统计一个pair的任一谓词一次都没有进入过top100的pair数量"
    )
    
    parser.add_argument(
        '--json_file',
        type=str,
        default='/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_2000.json',
        help='预测结果JSON文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='导出结果到指定JSON文件（可选）'
    )
    
    args = parser.parse_args()
    
    # 加载结果
    data = load_results(args.json_file)
    
    # 统计
    results = count_pairs_not_in_top100(data)
    
    if results is None:
        print("❌ 统计失败")
        return
    
    # 打印结果
    print_statistics(results)
    
    # 导出结果
    if args.output:
        export_results(results, args.output)
    
    print("\n✅ 统计完成！")


if __name__ == "__main__":
    main()

