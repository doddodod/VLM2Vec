#!/usr/bin/env python3
"""
检查 stage2 生成的 CoT 数据是否覆盖了 test_2000_images.json 中所有的 pair
"""

import json
from collections import defaultdict

# 文件路径
INPUT_DATA_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_2000_images.json"
STAGE2_OUTPUT_FILE = "/public/home/xiaojw2025/Data/stage2/stage2_generated_results.json"

def extract_pairs_from_input(input_file):
    """从输入文件中提取所有的 (image_id, subject, object) pair"""
    print(f"正在读取输入文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    all_pairs = set()
    image_pairs = defaultdict(set)  # image_id -> set of (subject, object)
    
    for img_data in input_data:
        image_id = img_data['image_id']
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
                
                # 使用字符串类型的 image_id 以匹配输出格式
                pair_key = (str(image_id), subject, object_name)
                all_pairs.add(pair_key)
                image_pairs[str(image_id)].add((subject, object_name))
    
    print(f"✓ 从输入文件中提取了 {len(all_pairs)} 个唯一的 pair")
    print(f"✓ 涉及 {len(image_pairs)} 张图片")
    
    return all_pairs, image_pairs

def extract_pairs_from_output(output_file):
    """从输出文件中提取所有的 (image_id, subject, object) pair"""
    print(f"\n正在读取输出文件: {output_file}")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    all_pairs = set()
    image_pairs = defaultdict(set)  # image_id -> set of (subject, object)
    
    results = output_data.get('results', [])
    
    for result in results:
        image_id = str(result['image_id'])  # 确保是字符串类型
        subject = result['subject']
        object_name = result['object']
        
        pair_key = (image_id, subject, object_name)
        all_pairs.add(pair_key)
        image_pairs[image_id].add((subject, object_name))
    
    print(f"✓ 从输出文件中提取了 {len(all_pairs)} 个唯一的 pair")
    print(f"✓ 涉及 {len(image_pairs)} 张图片")
    
    return all_pairs, image_pairs

def check_coverage():
    """检查覆盖率"""
    print("=" * 80)
    print("检查 Stage2 生成的 CoT 数据是否覆盖了所有 pair")
    print("=" * 80)
    
    # 提取输入文件中的所有 pair
    input_pairs, input_image_pairs = extract_pairs_from_input(INPUT_DATA_FILE)
    
    # 提取输出文件中的所有 pair
    output_pairs, output_image_pairs = extract_pairs_from_output(STAGE2_OUTPUT_FILE)
    
    # 找出遗漏的 pair
    missing_pairs = input_pairs - output_pairs
    extra_pairs = output_pairs - input_pairs
    
    print("\n" + "=" * 80)
    print("覆盖率统计")
    print("=" * 80)
    print(f"输入文件中的 pair 总数: {len(input_pairs)}")
    print(f"输出文件中的 pair 总数: {len(output_pairs)}")
    print(f"遗漏的 pair 数量: {len(missing_pairs)}")
    print(f"额外的 pair 数量: {len(extra_pairs)}")
    
    if len(input_pairs) > 0:
        coverage_rate = (len(input_pairs) - len(missing_pairs)) / len(input_pairs) * 100
        print(f"覆盖率: {coverage_rate:.2f}%")
    
    # 按图片分组显示遗漏的 pair
    if missing_pairs:
        print("\n" + "=" * 80)
        print("遗漏的 pair 详情（按图片分组）")
        print("=" * 80)
        
        missing_by_image = defaultdict(list)
        for image_id, subject, object_name in missing_pairs:
            missing_by_image[image_id].append((subject, object_name))
        
        print(f"\n共有 {len(missing_by_image)} 张图片存在遗漏的 pair:")
        
        # 显示前20个图片的遗漏情况
        for idx, (image_id, pairs) in enumerate(sorted(missing_by_image.items())[:20], 1):
            print(f"\n图片 {image_id} (遗漏 {len(pairs)} 个 pair):")
            for subject, object_name in pairs[:10]:  # 每个图片最多显示10个
                print(f"  - ({subject}, {object_name})")
            if len(pairs) > 10:
                print(f"  ... 还有 {len(pairs) - 10} 个 pair")
        
        if len(missing_by_image) > 20:
            print(f"\n... 还有 {len(missing_by_image) - 20} 张图片存在遗漏")
        
        # 保存遗漏的 pair 到文件
        missing_output_file = "/public/home/xiaojw2025/Data/stage2/missing_pairs.json"
        missing_data = {
            'total_missing': len(missing_pairs),
            'missing_by_image': {
                img_id: [{'subject': s, 'object': o} for s, o in pairs]
                for img_id, pairs in missing_by_image.items()
            }
        }
        with open(missing_output_file, 'w', encoding='utf-8') as f:
            json.dump(missing_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 遗漏的 pair 已保存到: {missing_output_file}")
    else:
        print("\n✅ 所有 pair 都已覆盖！")
    
    # 显示额外的 pair（如果存在）
    if extra_pairs:
        print("\n" + "=" * 80)
        print("额外的 pair（在输出中存在但输入中不存在）")
        print("=" * 80)
        print(f"共有 {len(extra_pairs)} 个额外的 pair")
        
        # 显示前20个
        for idx, (image_id, subject, object_name) in enumerate(sorted(extra_pairs)[:20], 1):
            print(f"  {idx}. 图片 {image_id}: ({subject}, {object_name})")
        
        if len(extra_pairs) > 20:
            print(f"  ... 还有 {len(extra_pairs) - 20} 个额外的 pair")
    
    # 图片级别的覆盖率统计
    print("\n" + "=" * 80)
    print("图片级别的覆盖率统计")
    print("=" * 80)
    
    input_images = set(input_image_pairs.keys())
    output_images = set(output_image_pairs.keys())
    
    missing_images = input_images - output_images
    images_with_missing_pairs = set()
    
    for image_id in input_images:
        input_set = input_image_pairs[image_id]
        output_set = output_image_pairs.get(image_id, set())
        if input_set - output_set:
            images_with_missing_pairs.add(image_id)
    
    print(f"输入文件中的图片数: {len(input_images)}")
    print(f"输出文件中的图片数: {len(output_images)}")
    print(f"完全缺失的图片数: {len(missing_images)}")
    print(f"存在遗漏 pair 的图片数: {len(images_with_missing_pairs)}")
    
    if missing_images:
        print(f"\n完全缺失的图片 ID (前20个):")
        for img_id in sorted(missing_images)[:20]:
            print(f"  - {img_id}")
        if len(missing_images) > 20:
            print(f"  ... 还有 {len(missing_images) - 20} 张图片")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_coverage()

