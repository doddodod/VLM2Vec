# Pipeline完整流程说明：所有配对的构建和处理

## 📋 整体架构

```
输入数据 → Stage2 (CoT生成) → Stage3 (关系预测) → 评估
```

---

## 🔍 Stage 1: 数据输入

### 输入文件格式
```json
{
  "image_id": 123,
  "image_path": "/path/to/image.jpg",
  "objects": [
    {"id": 1, "class_name": "person", "bbox": [x1, y1, x2, y2]},
    {"id": 2, "class_name": "car", "bbox": [x1, y1, x2, y2]},
    {"id": 3, "class_name": "road", "bbox": [x1, y1, x2, y2]}
  ],
  "relations": [
    {"subject_id": 1, "object_id": 2, "predicate": "riding"},
    {"subject_id": 2, "object_id": 3, "predicate": "on"}
  ]
}
```

### 关键信息
- **objects**: 图片中的所有物体（N个物体）
- **relations**: GT关系（Ground Truth），格式为 (subject_id, object_id, predicate)

---

## 🎯 Stage 2: CoT描述生成

### 2.1 配对生成
**对所有物体进行两两配对**（在Stage2中完成）

假设图片有N个物体，则生成 **N × (N-1)** 个配对：
- (object_1, object_2)
- (object_1, object_3)
- ...
- (object_N, object_{N-1})

### 2.2 CoT描述生成
对每个配对 `(subject, object)`：
1. 构建查询：使用原始query（包含bbox和object_ref）
2. 调用模型：生成CoT风格的描述文本
3. 保存结果：`{(image_id, subject, object): cot_description}`

### 2.3 Stage2输出格式
```json
{
  "results": [
    {
      "image_id": 123,
      "subject": "person",
      "object": "car",
      "stage2_generated_description": "The person is sitting on the car..."
    },
    ...
  ]
}
```

**重要**：Stage2只为**有GT关系的配对**生成CoT描述（或者所有配对，取决于Stage2的实现）

---

## 🚀 Stage 3: 关系预测（当前代码）

### 3.1 加载数据

#### 3.1.1 加载Stage2的CoT数据
```python
cot_map = load_stage2_cot_data(stage2_file)
# 格式: {(image_id, subject, object): cot_description}
```

#### 3.1.2 加载输入数据
```python
data = json.load(input_file)
# 包含所有图片的objects和relations
```

### 3.2 对每张图片的处理

#### 步骤1: 初始化
```python
objects = img_data['objects']  # 例如：10个物体
relations = img_data['relations']  # GT关系

# 创建映射
obj_dict = {obj['id']: obj for obj in objects}
gt_relations_map = {(subject_id, object_id): [predicate1, predicate2, ...]}
```

#### 步骤2: 生成所有配对
```python
object_ids = list(obj_dict.keys())  # [1, 2, 3, ..., 10]

# 对所有物体进行两两配对
for i, subject_id in enumerate(object_ids):
    for j, object_id in enumerate(object_ids):
        if i == j:
            continue  # 跳过自己和自己
        
        # 生成配对: (subject_id, object_id)
        # 例如：(1, 2), (1, 3), ..., (10, 9)
```

**配对总数**：N × (N-1) 个配对

#### 步骤3: 对每个配对的处理

##### 3.3.1 查找CoT描述
```python
subject_name = subject_obj['class_name']  # "person"
object_name = object_obj['class_name']    # "car"

# 尝试多种key格式匹配
key_formats = [
    (str(image_id), subject_name, object_name),
    (image_id, subject_name, object_name),
    ...
]

cot_description = None
for key_format in key_formats:
    if key_format in cot_map:
        cot_description = cot_map[key_format]
        break
```

**关键点**：
- 如果找到CoT描述 → 继续处理
- 如果**没有找到CoT描述** → **跳过该配对**（`missing_cot_count++`）

##### 3.3.2 关系预测
```python
# 调用predict_relation函数
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
```

**predict_relation内部流程**：

1. **构建查询文本**
   ```python
   query_text = cot_description.strip()
   
   # 如果use_original_query=True
   if use_original_query:
       coordinate_prefix = f"In the given image, the subject {subj_ref} is located at {subj_bbox_token},the object{obj_ref} is located at {obj_bbox_token}. Please describe..."
       query_text = f"{image_token}{coordinate_prefix}{query_text}"
   ```

2. **编码查询**
   ```python
   inputs = processor(text=query_text, images=image, return_tensors="pt")
   qry_output = model(qry=inputs)["qry_reps"]  # [1, hidden_dim]
   ```

3. **计算与50个谓词的相似度**
   ```python
   predicate_scores = []
   for predicate in PREDICATES:  # 50个固定谓词
       similarity = model.compute_similarity(qry_output, predicate_vector)
       predicate_scores.append({
           'predicate': predicate,
           'similarity': similarity.item()
       })
   ```
   
   **返回**：50个谓词及其相似度分数

##### 3.3.3 判断GT关系
```python
# 检查该配对是否有GT关系
has_gt = (subject_id, object_id) in gt_relations_map
gt_predicates = gt_relations_map.get((subject_id, object_id), [])

# 例如：
# has_gt = True
# gt_predicates = ["riding"]  # 可能有多个GT谓词
```

##### 3.3.4 生成候选列表
```python
# 将该配对的50个谓词候选加入候选池
for pred_score in predicate_scores:  # 50个谓词
    is_correct = False
    if has_gt and pred_score['predicate'] in gt_predicates:
        is_correct = True  # 预测的谓词匹配GT谓词
    
    # 计算relation_idx
    relation_idx = -1
    if is_correct and has_gt:
        relation_idx = relation_idx_start + idx  # >= 0
    
    image_candidates.append({
        'relation_idx': relation_idx,  # -1 或 >= 0
        'image_id': image_id,
        'subject': subject_obj['class_name'],
        'object': object_obj['class_name'],
        'gt_predicate': gt_predicates[0] if gt_predicates else None,
        'gt_predicates': gt_predicates,  # 所有GT谓词
        'predicted_predicate': pred_score['predicate'],  # 50个中的一个
        'similarity': pred_score['similarity'],
        'is_correct': is_correct,
        'has_gt': has_gt
    })
```

**关键点**：
- 每个配对生成 **50个候选**（对应50个固定谓词）
- 每个候选包含：预测的谓词、相似度、是否匹配GT等信息

### 3.3 候选列表结构

对于一张有10个物体的图片：
- **配对数**：10 × 9 = 90个配对
- **候选总数**：90 × 50 = 4500个候选（如果所有配对都有CoT描述）

**候选列表格式**：
```python
image_candidates = [
    # 配对1 (person, car) 的50个候选
    {'subject': 'person', 'object': 'car', 'predicted_predicate': 'riding', 'similarity': 0.95, 'has_gt': True, 'is_correct': True, 'relation_idx': 0},
    {'subject': 'person', 'object': 'car', 'predicted_predicate': 'near', 'similarity': 0.85, 'has_gt': True, 'is_correct': False, 'relation_idx': -1},
    ... (共50个)
    
    # 配对2 (person, road) 的50个候选
    {'subject': 'person', 'object': 'road', 'predicted_predicate': 'on', 'similarity': 0.78, 'has_gt': False, 'is_correct': False, 'relation_idx': -1},
    ... (共50个)
    
    # ... 其他88个配对
]
```

---

## 📊 Stage 4: 评估阶段

### 4.1 候选列表的保存

Stage3输出包含：
```json
{
  "per_image_top100_candidates": {
    "image_id": [
      // 每张图片的Top-100候选（按相似度排序）
    ]
  },
  "per_image_candidates": {
    // 所有候选（如果保存的话）
  }
}
```

### 4.2 召回率计算

#### 4.2.1 基于候选列表中的GT（evaluate_results.py）

```python
# 只统计候选列表中的GT关系
gt_relations = set()
for cand in candidates:
    if cand['relation_idx'] >= 0:  # 只统计relation_idx >= 0的
        gt_relations.add(cand['relation_idx'])

# 过滤no relation，取Top-K
non_bg_candidates = [c for c in candidates if c['predicted_predicate'] != 'no relation']
top_k = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)[:k]

# 统计召回
recalled = set()
for cand in top_k:
    if cand['relation_idx'] in gt_relations and cand['is_correct']:
        recalled.add(cand['relation_idx'])

recall = len(recalled) / len(gt_relations)
```

**特点**：
- 分母：候选列表中的GT关系数（`relation_idx >= 0`）
- 只评估进入候选列表的GT关系

#### 4.2.2 基于完整GT（evaluate_with_gt.py）

```python
# 从GT文件加载完整GT
full_gt_pairs = load_gt_data(gt_file)  # 所有GT关系对

# 取Top-K
top_k = sorted(candidates, key=lambda x: x['similarity'], reverse=True)[:k]

# 统计召回（检查Top-K中的预测是否在完整GT中）
recalled_pairs = set()
for cand in top_k:
    pair_key = (cand['subject'], cand['object'], cand['predicted_predicate'])
    if pair_key in full_gt_pairs:  # 检查是否在完整GT中
        recalled_pairs.add(pair_key)

recall = len(recalled_pairs) / len(full_gt_pairs)
```

**特点**：
- 分母：完整GT中的所有关系对
- 评估所有GT关系，包括未进入候选列表的

---

## 🔑 关键概念总结

### 1. 配对（Pair）
- **定义**：两个物体之间的配对 `(subject, object)`
- **数量**：N × (N-1) 个配对（N个物体）
- **处理**：每个配对需要CoT描述才能进行预测

### 2. 候选（Candidate）
- **定义**：一个配对的一个谓词预测
- **数量**：每个配对生成50个候选（对应50个固定谓词）
- **格式**：`(subject, object, predicted_predicate, similarity)`

### 3. 候选列表（Candidates List）
- **定义**：所有配对的所有候选的集合
- **数量**：配对数 × 50（如果所有配对都有CoT描述）
- **特点**：包含所有可能的预测，未排序

### 4. Top-K召回
- **定义**：从候选列表中取相似度最高的K个候选，统计其中有多少是GT关系
- **步骤**：
  1. 过滤no relation
  2. 按相似度排序
  3. 取Top-K
  4. 统计GT关系

### 5. GT关系的两种统计方式

#### 方式A：候选列表中的GT
- **条件**：`has_gt=True` 且 `relation_idx >= 0`
- **含义**：该配对的预测谓词恰好匹配GT谓词
- **数量**：≤ 完整GT数量

#### 方式B：完整GT
- **来源**：从GT文件直接读取
- **含义**：所有标注的GT关系
- **数量**：所有GT关系

---

## ⚠️ 重要注意事项

### 1. CoT描述缺失
- 如果配对没有CoT描述 → **跳过该配对**
- 该配对的所有50个候选都不会生成
- 对应的GT关系（如果有）不会出现在候选列表中

### 2. GT谓词不在50个固定谓词中
- 50个谓词是固定的列表
- 如果GT谓词不在列表中 → 该GT关系无法被预测
- 但配对仍然会生成50个候选（只是不包含GT谓词）

### 3. 一个配对可能有多个GT关系
```python
gt_relations_map[(subject_id, object_id)] = ["riding", "near"]
# 一个配对可能有多个GT谓词
```

### 4. relation_idx的含义
- `relation_idx >= 0`：预测的谓词匹配GT谓词，且该GT关系在候选列表中
- `relation_idx == -1`：预测的谓词不匹配GT谓词，或该配对没有GT关系

---

## 📈 数据流示例

假设一张图片：
- **物体数**：3个（person, car, road）
- **GT关系**：2个
  - (person, car, "riding")
  - (car, road, "on")

### Stage2输出
```
(person, car) → CoT描述1
(car, road) → CoT描述2
(person, road) → 无CoT描述（假设）
```

### Stage3处理

#### 配对1: (person, car)
- ✅ 有CoT描述 → 继续处理
- ✅ 有GT关系：["riding"]
- 生成50个候选：
  - (person, car, "riding", 0.95) → `relation_idx=0`, `is_correct=True`
  - (person, car, "near", 0.85) → `relation_idx=-1`, `is_correct=False`
  - ... (共50个)

#### 配对2: (car, road)
- ✅ 有CoT描述 → 继续处理
- ✅ 有GT关系：["on"]
- 生成50个候选：
  - (car, road, "on", 0.88) → `relation_idx=1`, `is_correct=True`
  - ... (共50个)

#### 配对3: (person, road)
- ❌ 无CoT描述 → **跳过**
- ❌ 无GT关系
- **不生成任何候选**

### 候选列表统计
- **配对数**：3 × 2 = 6个配对
- **处理的配对**：2个（有CoT描述的）
- **候选总数**：2 × 50 = 100个候选
- **候选列表中的GT**：2个（`relation_idx >= 0`）
- **完整GT**：2个

### 评估结果
- **基于候选列表中的GT**：分母 = 2
- **基于完整GT**：分母 = 2（如果所有GT关系都进入了候选列表）

---

## 🎯 总结

1. **配对生成**：对所有物体两两配对，生成 N×(N-1) 个配对
2. **CoT匹配**：查找每个配对的CoT描述，没有则跳过
3. **关系预测**：对每个有CoT的配对，预测50个谓词的相似度
4. **候选生成**：每个配对生成50个候选，加入候选列表
5. **评估**：从候选列表中取Top-K，统计GT关系的召回率

**关键点**：
- 候选列表包含所有配对的所有50个谓词预测
- 但只有有CoT描述的配对才会生成候选
- GT关系可能不在候选列表中（如果配对没有CoT描述或GT谓词不在50个固定谓词中）

