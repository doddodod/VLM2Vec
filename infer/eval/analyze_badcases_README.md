# Badcase分析脚本使用说明

## 功能概述

`analyze_badcases.py` 是一个详细的badcase分析工具，用于分析推理结果中每个GT pair的相似度排名情况。

### 主要功能

1. **GT谓词排名分析**：对每个GT pair，显示GT谓词在所有50个谓词中的相似度排名
2. **相似度分数显示**：显示GT谓词的相似度分数
3. **Badcase识别**：自动识别GT谓词排名较低的情况（可自定义阈值）
4. **详细排名信息**：显示Top-10和前20个谓词的详细排名
5. **按图片分析**：支持按图片ID查看详细分析
6. **结果导出**：支持导出详细分析结果到JSON文件

## 使用方法

### 1. 分析所有GT pair的排名情况

```bash
python analyze_badcases.py --json_file result_recall_20_all.json
```

这会显示：
- 各图片的统计信息（总pair数、Badcase数、正常case数、Badcase率）
- Badcase摘要（排名>20的情况）
- 前N个最严重的badcase

### 2. 分析指定图片

```bash
python /public/home/xiaojw2025/Workspace/VLM2Vec/embedding/eval/analyze_badcases.py --json_file result_recall_20_all.json --image_id 2317469
```

这会显示指定图片中所有GT pair的详细分析，包括：
- GT谓词的相似度分数
- GT谓词在所有谓词中的排名
- Top-10谓词排名
- 前20个谓词的详细排名

### 3. 自定义Badcase阈值

```bash
python analyze_badcases.py --json_file result_recall_20_all.json --rank_threshold 10
```

默认阈值是20，即排名>20认为是badcase。可以通过`--rank_threshold`参数自定义。

### 4. 导出详细分析结果

```bash
python analyze_badcases.py --json_file result_recall_20_all.json --export badcase_analysis.json
```

导出的JSON文件包含所有GT pair的详细分析结果，包括：
- GT谓词相似度
- GT谓词排名
- 所有谓词的排名列表
- Top-10谓词

### 5. 显示更多Badcase

```bash
python analyze_badcases.py --json_file result_recall_20_all.json --show_top_badcases 30
```

默认显示前20个badcase，可以通过`--show_top_badcases`参数调整。

## 输出说明

### 单个Pair分析输出

```
================================================================================
图片ID: 2317469
配对: person (ID:0) -> snow (ID:1)
GT谓词: walking in
================================================================================
GT谓词相似度: 0.271484
GT谓词排名: 2/51
✅ GT谓词排名较高（Top-10）

Top-10 谓词排名:
   1.    has                 : 0.277344
   2. ✅ walking in          : 0.271484
   3.    under               : 0.269531
   ...
```

说明：
- **GT谓词相似度**：GT谓词在所有50个谓词中的相似度分数
- **GT谓词排名**：GT谓词在所有谓词中的排名（从1开始）
- **Top-10谓词排名**：相似度最高的10个谓词及其分数
- **✅标记**：标记GT谓词在排名中的位置

### Badcase摘要输出

```
Badcase摘要（显示前N个）
================================================================================
1. 图片#2317469: person -> umbrella
   GT谓词: holding
   排名: 40/51
   相似度: 0.215820
   Top-3谓词: has(0.2832), walking in(0.2832), in front of(0.2734)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--json_file` | 预测结果JSON文件路径 | 必需 |
| `--image_id` | 指定要分析的图片ID（可选） | None |
| `--rank_threshold` | Badcase排名阈值 | 20 |
| `--export` | 导出详细分析结果到JSON文件 | None |
| `--show_top_badcases` | 显示前N个badcase | 20 |

## 使用场景

1. **Badcase分析**：快速识别GT谓词排名较低的pair，分析模型失败原因
2. **模型改进**：通过分析badcase，了解模型在哪些谓词上表现较差
3. **数据质量检查**：检查GT标注是否正确
4. **性能评估**：评估模型对不同谓词的预测能力

## 注意事项

1. 脚本需要从`per_image_all_candidates`或`per_image_top100_candidates`中获取候选数据
2. 如果结果文件中没有包含所有50个谓词的候选，排名可能不准确
3. 脚本支持使用物体ID区分同名物体，也支持向后兼容（使用类别名）

## 示例输出

### 成功案例（排名较高）

```
GT谓词排名: 2/51
✅ GT谓词排名较高（Top-10）
```

### Badcase（排名较低）

```
GT谓词排名: 40/51
❌ GT谓词排名较低（Top-40）
```

### 中等排名

```
GT谓词排名: 12/51
⚠️  GT谓词排名中等（Top-20）
```

