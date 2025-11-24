# Prompt示例：基于相对排名的CoT风格关系描述生成

本文档展示了一个实际的prompt示例，展示如何为图片中的一个pair生成CoT风格的详细关系描述。

## 示例场景

假设我们有一张图片，其中包含两个对象：
- **主体对象（Subject）**: `person`，位于bbox `[100, 150, 300, 450]`
- **客体对象（Object）**: `bicycle`，位于bbox `[200, 200, 400, 500]`

在Stage 1的top-100候选结果中，这个pair出现了以下候选谓词（按相似度从高到低排序，最多显示10个）：

| 相对排名 | Predicate |
|---------|-----------|
| 1       | riding    |
| 2       | on        |
| 3       | near      |
| 4       | with      |
| 5       | holding   |
| 6       | beside    |
| 7       | next to   |
| 8       | watching  |
| 9       | looking at|
| 10      | behind    |

## 实际输入Prompt

```
In this image, there are two objects:
- <|object_ref_start|>person<|object_ref_end|> at <|box_start|>(333, 333, 1000, 1000)<|box_end|>
- <|object_ref_start|>bicycle<|object_ref_end|> at <|box_start|>(666, 400, 1000, 1000)<|box_end|>

Stage 1 predicted candidate predicates for this pair (ranked by similarity, top candidates):
1. riding
2. on
3. near
4. with
5. holding
6. beside
7. next to
8. watching
9. looking at
10. behind

Based on the ranking information above, please provide a comprehensive and detailed description of the relationship between person and bicycle. Your description should:
1. Analyze the ranking positions of different candidate predicates
2. Consider the visual evidence in the image
3. Provide step-by-step reasoning (chain of thought) about why certain predicates are more likely than others
4. Give a thorough, detailed summary of the relationship that is as comprehensive as possible
5. Ensure your conclusions are well-reasoned and accurate

Please write a long, detailed description with clear reasoning steps.
```

## Prompt结构说明

### 1. 对象信息部分
```
In this image, there are two objects:
- <|object_ref_start|>person<|object_ref_end|> at <|box_start|>(333, 333, 1000, 1000)<|box_end|>
- <|object_ref_start|>bicycle<|object_ref_end|> at <|box_start|>(666, 400, 1000, 1000)<|box_end|>
```
- 使用特殊的token格式 `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>` 来标记对象和bbox
- bbox坐标已归一化到[0, 1000)范围

### 2. 候选谓词相对排名信息部分
```
Stage 1 predicted candidate predicates for this pair (ranked by similarity, top candidates):
1. riding
2. on
3. near
...
```
- 显示该pair的候选谓词的相对排名（1, 2, 3...）
- 最多显示10个候选谓词（按相似度从高到低排序）
- 不显示相似度分数，只显示相对排名
- 排名数字表示在该pair的候选谓词中的相对位置（1表示最可能，2表示次可能，以此类推）

### 3. 任务要求部分
要求模型：
1. **分析排序位置** - 理解不同候选谓词的排名意义
2. **考虑视觉证据** - 结合图片中的实际视觉信息
3. **提供CoT推理** - 逐步推理为什么某些谓词更可能
4. **给出详细总结** - 尽可能全面和详细的关系描述
5. **确保结论准确** - 推理过程要合理，结论要准确

## 预期输出格式

模型应该生成一个CoT风格的详细描述，例如：

```
Let me analyze the relationship between the person and bicycle step by step.

First, I observe the ranking positions: "riding" ranks highest at position 1, which suggests it is the most likely predicate for this pair. This indicates that the embedding model strongly associates this spatial configuration with a riding relationship.

Looking at the visual evidence in the image, I can see that:
- The person's position relative to the bicycle suggests an active interaction
- The spatial arrangement indicates the person is likely positioned on or above the bicycle
- The relative sizes and positions align with a typical riding scenario

Step-by-step reasoning:
1. The top-ranked predicate "riding" (position 1) is the most likely because:
   - It ranks highest among all candidate predicates for this pair
   - The spatial configuration supports this interpretation
   - This is a common relationship pattern between person and bicycle
   - The visual evidence aligns with an active riding interaction

2. The second-ranked "on" (position 2) is also plausible but less specific:
   - It captures the spatial relationship but lacks the dynamic aspect
   - The lower rank suggests it's a less precise match than "riding"
   - It could be a valid alternative interpretation

3. Other predicates like "near" (position 3), "beside" (position 6), "next to" (position 7) suggest spatial proximity but don't capture the specific interaction type. These lower-ranked predicates indicate less likely relationships.

4. Predicates like "watching" (position 8) and "looking at" (position 9) seem less likely given the spatial configuration, as they imply a different type of interaction that doesn't match the visual evidence.

Conclusion: Based on the ranking information and visual evidence, the most accurate description of the relationship is that the person is riding the bicycle. The top ranking of "riding" (position 1), combined with the spatial arrangement visible in the image, strongly support this interpretation. The relationship involves active physical interaction where the person is positioned on the bicycle in a manner consistent with riding, rather than merely being near, beside, or watching it. The relative ranking of other predicates helps confirm that "riding" is the most appropriate description, as alternatives rank lower and are less consistent with the visual evidence.
```

## 关键特点

1. **每pair一次调用**: 每个图片的每个pair都会单独调用一次prompt
2. **相对排名信息**: prompt中显示每个候选谓词的相对排名（1, 2, 3...），不显示绝对排序或相似度分数
3. **最多10个候选**: 只显示该pair中相似度最高的10个候选谓词
4. **CoT风格**: 要求模型进行逐步推理，体现思考过程
5. **详细描述**: 要求生成尽可能长和详细的关系描述
6. **基于相对排名**: 模型需要基于相对排名来分析哪些谓词更可能

## 注意事项

- bbox坐标会被归一化到[0, 1000)范围
- 只包含在top-100中出现的候选谓词（排除"no relation"）
- 相对排名从1开始计数（1表示该pair中最可能的谓词，2表示次可能的，以此类推）
- 最多显示10个候选谓词，按相似度从高到低排序
- 不显示相似度分数，只显示相对排名
- 不显示绝对排序（在top-100中的位置），只显示相对排名（在该pair候选中的位置）

