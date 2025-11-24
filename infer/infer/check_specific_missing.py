#!/usr/bin/env python3
"""
检查 Stage1 中存在但 Stage2 中缺失的 2 个 pair 的具体情况
"""

import json

# 文件路径
STAGE1_RESULT_FILE = "/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_2000.json"
STAGE2_OUTPUT_FILE = "/public/home/xiaojw2025/Data/stage2/stage2_generated_results.json"

# 需要检查的 pair
target_pairs = [
    ("2341571", "head", "cow"),
    ("2342806", "tail", "horse")
]

def check_pairs_in_stage1():
    """检查这些 pair 在 Stage1 中的情况"""
    print("检查 Stage1 中的情况:")
    print("=" * 80)
    
    with open(STAGE1_RESULT_FILE, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    
    per_image_top100 = stage1_data.get('per_image_top100_candidates', {})
    
    for image_id, subject, object_name in target_pairs:
        print(f"\n图片 {image_id}, pair: ({subject}, {object_name})")
        
        # 尝试不同的 image_id 格式
        candidates = None
        if image_id in per_image_top100:
            candidates = per_image_top100[image_id]
        elif int(image_id) in per_image_top100:
            candidates = per_image_top100[int(image_id)]
        
        if candidates:
            print(f"  ✓ 在 Stage1 的 top100 中找到 {len(candidates)} 个候选")
            
            # 查找这个 pair 的候选
            pair_candidates = []
            for candidate in candidates:
                if candidate.get('subject') == subject and candidate.get('object') == object_name:
                    pair_candidates.append(candidate)
            
            if pair_candidates:
                print(f"  ✓ 找到 {len(pair_candidates)} 个匹配的候选:")
                for idx, cand in enumerate(pair_candidates[:5], 1):
                    predicate = cand.get('predicted_predicate', 'no relation')
                    similarity = cand.get('similarity', 0)
                    has_gt = cand.get('has_gt', False)
                    print(f"    {idx}. predicate: {predicate}, similarity: {similarity:.4f}, has_gt: {has_gt}")
            else:
                print(f"  ✗ 没有找到匹配的候选（subject/object 不匹配）")
        else:
            print(f"  ✗ 图片 {image_id} 不在 Stage1 结果中")

def check_pairs_in_stage2():
    """检查这些 pair 在 Stage2 中的情况"""
    print("\n\n检查 Stage2 中的情况:")
    print("=" * 80)
    
    with open(STAGE2_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)
    
    results = stage2_data.get('results', [])
    
    for image_id, subject, object_name in target_pairs:
        print(f"\n图片 {image_id}, pair: ({subject}, {object_name})")
        
        # 查找这个 pair 的结果
        found = False
        for result in results:
            if (str(result['image_id']) == image_id and 
                result['subject'] == subject and 
                result['object'] == object_name):
                found = True
                print(f"  ✓ 在 Stage2 结果中找到")
                print(f"    ranked_predicates: {result.get('ranked_predicates', [])[:5]}")
                print(f"    has_gt: {result.get('has_gt', False)}")
                break
        
        if not found:
            print(f"  ✗ 在 Stage2 结果中未找到")

if __name__ == "__main__":
    check_pairs_in_stage1()
    check_pairs_in_stage2()

