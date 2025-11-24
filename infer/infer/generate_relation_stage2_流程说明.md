# 第二阶段 Relation 生成 Pipeline 详细流程说明

## 📋 目录

1. [概述](#概述)
2. [整体架构](#整体架构)
3. [数据流](#数据流)
4. [详细处理流程](#详细处理流程)
5. [关键数据结构](#关键数据结构)
6. [多GPU多Worker机制](#多gpu多worker机制)
7. [批量生成优化](#批量生成优化)
8. [断点续传机制](#断点续传机制)

---

## 概述

### 功能目标
基于第一阶段（embedding方法）预测的高置信度relation结果，使用生成模型（Qwen3-VL）生成详细的relation描述文本。

### 输入输出
- **输入1**: 第一阶段结果文件（`recall_results_*.json`）
  - 包含每张图片的Top-100候选relation
  - 每个候选包含：subject, object, predicted_predicate, similarity等
  
- **输入2**: 原始输入数据文件（`test_2000_images.json`）
  - 包含图片路径、物体信息（bbox等）
  
- **输出**: 第二阶段生成结果（`stage2_generated_results.json`）
  - 包含每个配对的第一阶段top-K relation
  - 以及对应的详细生成描述

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      main() 函数                             │
│  1. 解析命令行参数                                           │
│  2. 加载数据（第一阶段结果 + 原始输入数据）                  │
│  3. 准备推理数据（prepare_data_for_inference）              │
│  4. 选择推理模式（单GPU / 多GPU）                            │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼────────┐         ┌───────▼────────┐
        │   单GPU模式     │         │   多GPU模式     │
        │                 │         │                 │
        │ 直接调用生成函数 │         │ 启动多个进程    │
        │                 │         │ inference_on_gpu│
        └─────────────────┘         └─────────────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                            ┌───────▼───────┐  ┌───────▼───────┐
                            │  inference_on_gpu│  │  inference_on_gpu│
                            │  (GPU 0)        │  │  (GPU 1)        │
                            └─────────────────┘  └─────────────────┘
                                    │                   │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │   merge_results   │
                                    │   合并所有结果     │
                                    └───────────────────┘
```

---

## 数据流

### 1. 数据加载阶段

```
第一阶段结果文件 (JSON)
├── per_image_top100_candidates: {
│     "2343729": [
│       {
│         "subject": "windshield",
│         "object": "train",
│         "predicted_predicate": "on",
│         "similarity": 0.85,
│         "has_gt": true,
│         "gt_predicates": ["on"]
│       },
│       ... (最多100个)
│     ]
│   }
└── ...

原始输入数据文件 (JSON)
├── [
│   {
│     "image_id": 2343729,
│     "image_path": "/path/to/image.jpg",
│     "objects": [
│       {
│         "id": 1,
│         "class_name": "windshield",
│         "bbox": [100, 200, 300, 400]
│       },
│       ...
│     ],
│     "relations": [...]
│   },
│   ...
│ ]
```

### 2. 数据准备阶段（prepare_data_for_inference）

**处理步骤：**

1. **遍历每张图片的Top-100候选**
   ```python
   for image_id, top100_candidates in per_image_top100.items():
   ```

2. **匹配原始数据**
   - 通过image_id在input_data_map中查找对应的图片数据
   - 支持字符串和整数类型的image_id匹配

3. **按配对分组**
   - 将同一张图片的多个候选按(subject, object)配对分组
   - 过滤掉"no relation"的候选

4. **选择Top-K**
   - 对每个配对的候选按similarity排序
   - 选择top-K个（默认10个）高置信度relation

5. **构建pair_data**
   - 为每个配对创建一个pair_data对象
   - 包含subject_obj、object_obj、stage1_top_k等信息

**输出：`all_pairs` 列表**
```python
[
  {
    'image_id': '2343729',
    'image_path': '/path/to/image.jpg',
    'subject': 'windshield',
    'object': 'train',
    'subject_obj': {'class_name': 'windshield', 'bbox': [...], ...},
    'object_obj': {'class_name': 'train', 'bbox': [...], ...},
    'stage1_top_k': [
      {'predicate': 'on', 'similarity': 0.85},
      {'predicate': 'above', 'similarity': 0.72},
      ... (最多10个)
    ],
    'has_gt': True,
    'gt_predicates': ['on']
  },
  ... (所有配对的列表)
]
```

---

## 详细处理流程

### 阶段1: 初始化（main函数）

```python
main()
├── 解析命令行参数
│   ├── --stage1_result: 第一阶段结果文件
│   ├── --input_data: 原始输入数据文件
│   ├── --output: 输出文件路径
│   ├── --model_path: 生成模型路径
│   ├── --num_gpus: GPU数量
│   ├── --batch_size: 批量大小
│   ├── --workers_per_gpu: 每个GPU的worker数
│   └── --top_k: Top-K relation数量
│
├── 加载第一阶段结果
│   └── stage1_data = json.load(stage1_result_file)
│
├── 加载原始输入数据
│   └── input_data = json.load(input_data_file)
│
└── 创建image_data_map（支持字符串/整数类型匹配）
```

### 阶段2: 数据准备（prepare_data_for_inference函数）

```python
prepare_data_for_inference(stage1_data, image_data_map)
│
├── 获取per_image_top100_candidates
│   └── per_image_top100 = stage1_data['per_image_top100_candidates']
│
├── 遍历每张图片
│   for image_id, top100_candidates in per_image_top100.items():
│   │
│   ├── 匹配原始数据（支持类型转换）
│   │   └── img_data = image_data_map[image_id]
│   │
│   ├── 创建物体映射
│   │   └── obj_dict = {obj['class_name']: obj for obj in objects}
│   │
│   ├── 按配对分组候选
│   │   └── pair_candidates = group_by_pair(top100_candidates)
│   │
│   └── 对每个配对
│       for (subject, object), candidates in pair_candidates.items():
│       │
│       ├── 按相似度排序，取Top-K
│       │   └── top_k_candidates = sorted(candidates)[:TOP_K_RELATIONS]
│       │
│       └── 构建pair_data
│           └── all_pairs.append({
│                   'image_id': ...,
│                   'subject': ...,
│                   'object': ...,
│                   'subject_obj': ...,
│                   'object_obj': ...,
│                   'stage1_top_k': [...],
│                   ...
│               })
│
└── 返回 all_pairs
```

### 阶段3: 推理模式选择

#### 3.1 单GPU模式

```python
if args.num_gpus == 1:
    ├── 加载生成模型
    │   ├── GenModelClass = get_generation_model_class(model_path)
    │   ├── processor = AutoProcessor.from_pretrained(...)
    │   └── model = GenModelClass.from_pretrained(...)
    │
    ├── 检查断点续传
    │   └── processed_pairs = get_processed_pairs(existing_results)
    │
    ├── 过滤未处理配对
    │   └── unprocessed_pairs = filter_unprocessed(all_pairs, processed_pairs)
    │
    └── 逐个处理配对
        for pair_data in unprocessed_pairs:
            ├── 准备generation_tasks
            │   └── 为每个stage1_top_k创建task
            │
            ├── 批量生成
            │   └── stage2_results = generate_relations_batch(...)
            │
            └── 保存结果
```

#### 3.2 多GPU模式

```python
else:  # 多GPU模式
    ├── 检查最终合并结果的断点续传
    │   └── 过滤已处理的配对
    │
    ├── 分割数据到GPU
    │   └── data_chunks = split_data(all_pairs, num_gpus)
    │
    ├── 如果workers_per_gpu > 1
    │   ├── 计算每个worker的显存限制
    │   └── 进一步分割数据到worker
    │       └── worker_chunks = split_data(gpu_data, workers_per_gpu)
    │
    ├── 启动多个进程
    │   for gpu_id, worker_id, chunk in zip(...):
    │       └── mp.Process(target=inference_on_gpu, ...)
    │
    ├── 等待所有进程完成
    │   └── for p in processes: p.join()
    │
    └── 合并结果
        └── merge_results(...)
```

### 阶段4: GPU推理（inference_on_gpu函数）

```python
inference_on_gpu(gpu_id, data_chunk, model_path, ...)
│
├── 设置GPU设备
│   └── torch.cuda.set_device(gpu_id)
│
├── 加载模型
│   ├── GenModelClass = get_generation_model_class(model_path)
│   ├── config = AutoConfig.from_pretrained(...)
│   ├── model = GenModelClass.from_pretrained(..., device_map=f"cuda:{gpu_id}")
│   └── processor = AutoProcessor.from_pretrained(...)
│
├── 检查断点续传
│   ├── existing_results = load_existing_results(gpu_output_path)
│   └── processed_pairs = get_processed_pairs(existing_results)
│
├── 过滤未处理配对
│   └── unprocessed_chunk = filter_unprocessed(data_chunk, processed_pairs)
│
└── 处理每个配对
    for pair_data in tqdm(unprocessed_chunk):
        │
        ├── 打开图片，获取尺寸
        │   └── original_width, original_height = Image.open(image_path).size
        │
        ├── 准备批量生成任务
        │   for stage1_item in pair_data['stage1_top_k']:
        │       └── generation_tasks.append({
        │               'subject_obj': pair_data['subject_obj'],
        │               'object_obj': pair_data['object_obj'],
        │               'top_predicate': stage1_item['predicate'],
        │               'similarity': stage1_item['similarity']
        │           })
        │
        ├── 批量生成
        │   └── stage2_results = generate_relations_batch(
        │           model, processor, image_path, generation_tasks,
        │           original_width, original_height, batch_size
        │       )
        │
        ├── 保存结果
        │   └── result = {
        │           'image_id': ...,
        │           'subject': ...,
        │           'object': ...,
        │           'stage1_top_k': ...,
        │           'stage2_generated': stage2_results,
        │           ...
        │       }
        │
        ├── 定期保存（每SAVE_INTERVAL个配对）
        │   └── 保存到临时文件
        │
        └── 定期清理显存（每MEMORY_CLEANUP_INTERVAL个配对）
            └── torch.cuda.empty_cache()
```

### 阶段5: 批量生成（generate_relations_batch函数）

```python
generate_relations_batch(model, processor, image_path, generation_tasks, ...)
│
├── 可选：预处理图像（图像缓存优化）
│   if use_image_cache:
│       └── cached_pixel_values = processor(images=[image])['pixel_values']
│
├── 分批处理
│   for i in range(0, len(generation_tasks), batch_size):
│       │
│       ├── 获取当前batch
│       │   └── batch = generation_tasks[i:i+batch_size]
│       │
│       ├── 批量构建prompts
│       │   for task in batch:
│       │       ├── prompt_text = build_prompt(
│       │       │       task['subject_obj'], task['object_obj'],
│       │       │       task['top_predicate'], ...
│       │       │   )
│       │       ├── conversation = [{"role": "user", "content": [...]}]
│       │       └── text_prompt = processor.apply_chat_template(...)
│       │
│       ├── 批量处理输入
│       │   if use_image_cache:
│       │       └── 复用缓存的图像特征
│       │   else:
│       │       └── inputs = processor(text=text_prompts, images=[image]*batch_len)
│       │
│       ├── 批量生成
│       │   └── generated_ids = model.generate(**inputs, ...)
│       │
│       ├── 批量解码
│       │   for gen_id in generated_ids:
│       │       └── generated_text = processor.tokenizer.decode(...)
│       │
│       └── 收集结果
│           └── all_results.append({
│                   'predicate': ...,
│                   'similarity': ...,
│                   'generated_description': generated_text
│               })
│
└── 返回 all_results
```

---

## 关键数据结构

### 1. pair_data（配对数据）

```python
{
    'image_id': str/int,           # 图片ID
    'image_path': str,             # 图片路径
    'subject': str,                # 主体名称
    'object': str,                 # 客体名称
    'subject_obj': {               # 主体对象完整信息
        'class_name': str,
        'bbox': [x1, y1, x2, y2],
        'id': int,
        ...
    },
    'object_obj': {                # 客体对象完整信息
        'class_name': str,
        'bbox': [x1, y1, x2, y2],
        'id': int,
        ...
    },
    'stage1_top_k': [              # 第一阶段Top-K个relation
        {
            'predicate': str,       # 谓词
            'similarity': float     # 相似度
        },
        ...
    ],
    'has_gt': bool,                # 是否有GT关系
    'gt_predicates': [str, ...]    # GT谓词列表
}
```

### 2. generation_task（生成任务）

```python
{
    'subject_obj': {...},          # 主体对象信息
    'object_obj': {...},          # 客体对象信息
    'top_predicate': str,         # 谓词
    'similarity': float           # 相似度
}
```

### 3. stage2_result（生成结果）

```python
{
    'predicate': str,                    # 谓词
    'similarity': float,                 # 相似度
    'generated_description': str         # 生成的详细描述文本
}
```

### 4. 最终输出结果

```python
{
    'summary': {
        'total_pairs': int,
        'total_images': int,
        'top_k_relations': int,
        'generation_max_tokens': int,
        'generation_temperature': float,
        'num_gpus': int,
        'workers_per_gpu': int
    },
    'results': [
        {
            'image_id': str/int,
            'subject': str,
            'object': str,
            'stage1_top_k': [...],
            'stage2_generated': [
                {
                    'predicate': str,
                    'similarity': float,
                    'generated_description': str
                },
                ...
            ],
            'has_gt': bool,
            'gt_predicates': [str, ...]
        },
        ...
    ]
}
```

---

## 多GPU多Worker机制

### 数据分割策略

```
总配对数据 (all_pairs)
    │
    ├── 第一层分割：按GPU数量
    │   ├── GPU 0: data_chunks[0]
    │   ├── GPU 1: data_chunks[1]
    │   ├── GPU 2: data_chunks[2]
    │   └── GPU 3: data_chunks[3]
    │
    └── 第二层分割（如果workers_per_gpu > 1）：按Worker数量
        ├── GPU 0
        │   ├── Worker 0: worker_chunks[0]
        │   └── Worker 1: worker_chunks[1]
        ├── GPU 1
        │   ├── Worker 0: worker_chunks[2]
        │   └── Worker 1: worker_chunks[3]
        ...
```

### 进程启动

```python
# 使用multiprocessing启动多个进程
for gpu_id, worker_id, chunk in zip(worker_gpu_ids, worker_ids, worker_chunks):
    p = mp.Process(
        target=inference_on_gpu,
        args=(gpu_id, chunk, model_path, output_prefix,
              shared_stats, batch_size, worker_id, max_memory_per_worker)
    )
    p.start()
    processes.append(p)

# 等待所有进程完成
for p in processes:
    p.join()
```

### 显存管理（多Worker模式）

- 每个GPU的显存被平均分配给多个worker
- 每个worker使用 `max_memory` 参数限制显存使用
- 计算公式：`max_memory_per_worker = gpu_memory_mb * 0.8 / workers_per_gpu`

### 结果合并

```python
merge_results(output_prefix, num_gpus, final_output_path, ...)
│
├── 读取每个GPU/Worker的结果文件
│   for gpu_id in range(num_gpus):
│       if workers_per_gpu > 1:
│           for worker_id in range(workers_per_gpu):
│               └── 读取 gpu{gpu_id}_worker{worker_id}.json
│       else:
│           └── 读取 gpu{gpu_id}.json
│
├── 去重合并（基于image_id, subject, object）
│   └── 使用processed_pairs集合避免重复
│
└── 保存最终结果
    └── 写入 final_output_path
```

---

## 批量生成优化

### Batch Processing流程

```
generation_tasks (一个配对的所有Top-K relation)
    │
    ├── Batch 1: tasks[0:batch_size]
    │   ├── 构建prompts
    │   ├── 处理输入
    │   ├── 批量生成
    │   └── 解码结果
    │
    ├── Batch 2: tasks[batch_size:2*batch_size]
    │   └── ...
    │
    └── Batch N: tasks[(N-1)*batch_size:N*batch_size]
        └── ...
```

### 图像缓存优化（可选）

如果 `USE_IMAGE_CACHE = True`：

1. **预处理阶段**：只处理图像一次，缓存图像特征
   ```python
   cached_pixel_values = processor(images=[image])['pixel_values']
   ```

2. **批量处理阶段**：只处理文本，复用缓存的图像特征
   ```python
   text_inputs = processor(text=text_prompts, images=None)
   text_inputs['pixel_values'] = cached_pixel_values.repeat(batch_len, 1, 1, 1)
   ```

**优势**：避免重复编码图像，提高处理速度（1.5-2倍加速）

---

## 断点续传机制

### 单GPU模式

1. **检查已存在结果**
   ```python
   existing_results = load_existing_results(output_path)
   processed_pairs = get_processed_pairs(existing_results)
   ```

2. **过滤未处理配对**
   ```python
   unprocessed_pairs = [p for p in all_pairs 
                        if (p['image_id'], p['subject'], p['object']) 
                        not in processed_pairs]
   ```

3. **继续处理**
   - 只处理未完成的配对
   - 结果追加到已存在结果中

### 多GPU模式

1. **检查最终合并结果**
   - 如果最终输出文件存在，从中提取已处理的配对
   - 过滤掉已处理的配对

2. **检查每个GPU/Worker的结果**
   - 每个GPU/Worker独立检查自己的结果文件
   - 只处理未完成的配对

3. **定期保存**
   - 每处理 `SAVE_INTERVAL` 个配对保存一次
   - 保存到临时文件，防止意外中断导致数据丢失

### 配对唯一标识

使用三元组作为唯一标识：
```python
pair_key = (image_id, subject, object)
```

---

## Prompt构建

### Prompt格式

```
这是一张图，物体<|object_ref_start|>{subject_name}<|object_ref_end|>和
<|object_ref_start|>{object_name}<|object_ref_end|>在位置
({subj_center_x}, {subj_center_y})和({obj_center_x}, {obj_center_y})，
<|object_ref_start|>{subject_name}<|object_ref_end|>位于
<|box_start|>({subject_bbox_str})<|box_end|>，
<|object_ref_start|>{object_name}<|object_ref_end|>位于
<|box_start|>({object_bbox_str})<|box_end|>，
{subject_name} is roughly {top_predicate} {object_name}，
请给出{subject_name}和{object_name}细节的相对关系
```

### Bbox归一化

- 将原始坐标归一化到[0, 1000)范围
- 格式：`"{x1}, {y1}, {x2}, {y2}"`

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_K_RELATIONS` | 10 | 每个配对选择的高置信度relation数量 |
| `BATCH_SIZE` | 8 | 批量推理的batch size |
| `MAX_NEW_TOKENS` | 512 | 生成的最大token数 |
| `TEMPERATURE` | 0.1 | 生成温度（越低越确定） |
| `SAVE_INTERVAL` | 50 | 每处理多少个配对保存一次 |
| `MEMORY_CLEANUP_INTERVAL` | 20 | 每处理多少个配对清理一次显存 |
| `USE_IMAGE_CACHE` | False | 是否使用图像缓存优化 |

---

## 执行示例

### 单GPU模式
```bash
python generate_relation_stage2.py \
    --num_gpus 1 \
    --batch_size 8 \
    --top_k 10
```

### 多GPU多Worker模式
```bash
python generate_relation_stage2.py \
    --num_gpus 4 \
    --workers_per_gpu 2 \
    --batch_size 6 \
    --top_k 10
```

---

## 总结

整个Pipeline的核心流程：

1. **数据加载** → 加载第一阶段结果和原始输入数据
2. **数据准备** → 匹配数据、分组、选择Top-K，构建pair_data
3. **推理执行** → 单GPU或多GPU模式，批量生成详细描述
4. **结果保存** → 定期保存、断点续传、最终合并

关键优化：
- ✅ 批量生成（batch processing）
- ✅ 多GPU多Worker并行
- ✅ 图像缓存优化（可选）
- ✅ 断点续传
- ✅ 定期保存和显存清理

