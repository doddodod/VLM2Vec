#!/usr/bin/env python3
"""
分析遗漏的 pair 的原因：检查 Stage1 结果中是否包含这些 pair
"""

import json
from collections import defaultdict

# 文件路径
INPUT_DATA_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_2000_images.json"
STAGE1_RESULT_FILE = "/public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_2000.json"
STAGE2_OUTPUT_FILE = "/public/home/xiaojw2025/Data/stage2/stage2_generated_results.json"
MISSING_PAIRS_FILE = "/public/home/xiaojw2025/Data/stage2/missing_pairs.json"

def extract_pairs_from_input(input_file):
    """从输入文件中提取所有的 (image_id, subject, object) pair"""
    print(f"正在读取输入文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    all_pairs = set()
    
    for img_data in input_data:
        image_id = str(img_data['image_id'])
        objects = img_data['objects']
        relations = img_data.get('relations', [])
        
        # 创建 id -> class_name 的映射
        id_to_class = {obj['id']: obj['class_name'] for obj in objects}
        
        # 提取所有的 pair
        for relation in relations:
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            
            if subject_id in id_to_class and object_id in id_to_class:
                subject = id_to_class[subject_id]
                object_name = id_to_class[object_id]
                
                pair_key = (image_id, subject, object_name)
                all_pairs.add(pair_key)
    
    return all_pairs

def extract_pairs_from_stage1(stage1_file):
    """从 Stage1 结果中提取所有的 (image_id, subject, object) pair"""
    print(f"\n正在读取 Stage1 结果文件: {stage1_file}")
    
    with open(stage1_file, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    
    all_pairs = set()
    per_image_top100 = stage1_data.get('per_image_top100_candidates', {})
    
    for image_id, top100_candidates in per_image_top100.items():
        image_id_str = str(image_id)
        for candidate in top100_candidates:
            subject = candidate.get('subject')
            object_name = candidate.get('object')
            if subject and object_name:
                pair_key = (image_id_str, subject, object_name)
                all_pairs.add(pair_key)
    
    print(f"✓ Stage1 结果中包含 {len(all_pairs)} 个唯一的 pair")
    return all_pairs

def extract_pairs_from_stage2(stage2_file):
    """从 Stage2 结果中提取所有的 (image_id, subject, object) pair"""
    print(f"\n正在读取 Stage2 结果文件: {stage2_file}")
    
    with open(stage2_file, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)
    
    all_pairs = set()
    results = stage2_data.get('results', [])
    
    for result in results:
        image_id = str(result['image_id'])
        subject = result['subject']
        object_name = result['object']
        
        pair_key = (image_id, subject, object_name)
        all_pairs.add(pair_key)
    
    print(f"✓ Stage2 结果中包含 {len(all_pairs)} 个唯一的 pair")
    return all_pairs

def analyze_missing_pairs():
    """分析遗漏的 pair"""
    print("=" * 80)
    print("分析遗漏的 pair 的原因")
    print("=" * 80)
    
    # 提取所有 pair
    input_pairs = extract_pairs_from_input(INPUT_DATA_FILE)
    stage1_pairs = extract_pairs_from_stage1(STAGE1_RESULT_FILE)
    stage2_pairs = extract_pairs_from_stage2(STAGE2_OUTPUT_FILE)
    
    # 找出遗漏的 pair
    missing_from_stage2 = input_pairs - stage2_pairs
    missing_from_stage1 = input_pairs - stage1_pairs
    
    print("\n" + "=" * 80)
    print("分析结果")
    print("=" * 80)
    print(f"输入文件中的 pair 总数: {len(input_pairs)}")
    print(f"Stage1 结果中的 pair 总数: {len(stage1_pairs)}")
    print(f"Stage2 结果中的 pair 总数: {len(stage2_pairs)}")
    print(f"\n在 Stage1 中缺失的 pair 数: {len(missing_from_stage1)}")
    print(f"在 Stage2 中缺失的 pair 数: {len(missing_from_stage2)}")
    
    # 检查遗漏的 pair 是否在 Stage1 中
    missing_in_stage1 = missing_from_stage2 & missing_from_stage1
    missing_only_in_stage2 = missing_from_stage2 - missing_from_stage1
    
    print(f"\n既不在 Stage1 也不在 Stage2 的 pair 数: {len(missing_in_stage1)}")
    print(f"在 Stage1 中但不在 Stage2 中的 pair 数: {len(missing_only_in_stage2)}")
    
    if missing_only_in_stage2:
        print(f"\n⚠️  发现 {len(missing_only_in_stage2)} 个 pair 在 Stage1 中存在但 Stage2 中缺失！")
        print("这些 pair 应该被 Stage2 处理但没有被处理。")
        
        # 按图片分组显示
        missing_by_image = defaultdict(list)
        for image_id, subject, object_name in sorted(missing_only_in_stage2):
            missing_by_image[image_id].append((subject, object_name))
        
        print(f"\n前20个图片的遗漏情况:")
        for idx, (image_id, pairs) in enumerate(sorted(missing_by_image.items())[:20], 1):
            print(f"\n图片 {image_id} (遗漏 {len(pairs)} 个 pair):")
            for subject, object_name in pairs[:5]:
                print(f"  - ({subject}, {object_name})")
            if len(pairs) > 5:
                print(f"  ... 还有 {len(pairs) - 5} 个 pair")
    else:
        print("\n✅ 所有在 Stage1 中的 pair 都已被 Stage2 处理！")
        print("遗漏的 pair 是因为它们在 Stage1 中就不存在（没有进入 top100）。")
    
    # 统计覆盖率
    if len(input_pairs) > 0:
        stage1_coverage = (len(input_pairs) - len(missing_from_stage1)) / len(input_pairs) * 100
        stage2_coverage = (len(input_pairs) - len(missing_from_stage2)) / len(input_pairs) * 100
        print(f"\nStage1 覆盖率: {stage1_coverage:.2f}%")
        print(f"Stage2 覆盖率: {stage2_coverage:.2f}%")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_missing_pairs()

