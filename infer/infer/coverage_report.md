# Stage2 CoT 数据覆盖率检查报告

## 检查结果总结

### 覆盖率统计
- **输入文件中的 pair 总数**: 9,911
- **Stage2 生成的 pair 总数**: 31,738
- **遗漏的 pair 数量**: 4,524
- **覆盖率**: 54.35%

### 遗漏原因分析

#### 1. 主要遗漏原因（4,522 个 pair）
这些 pair 在 **Stage1 中就不存在**（没有进入 top100 候选），因此 Stage2 无法处理它们。

**原因**: Stage1 只保留了每个图片的 top100 候选 pair，如果某个 pair 的相似度排名不在前100，就不会出现在 Stage1 的结果中。

#### 2. 代码逻辑遗漏（2 个 pair）
有 2 个 pair 在 Stage1 中存在但被 Stage2 代码过滤掉了：

1. 图片 `2341571`: pair `(head, cow)`
2. 图片 `2342806`: pair `(tail, horse)`

**原因**: 这两个 pair 的所有候选的 `predicted_predicate` 都是 `"no relation"`，而代码在 `prepare_data_for_inference` 函数中（第766行）会过滤掉 `predicate == 'no relation'` 的候选：

```python
if predicate != 'no relation':
    pair_candidates[pair_key].append({
        'predicate': predicate,
        'similarity': candidate.get('similarity', 0)
    })
```

然后在第777行，如果 `candidates` 列表为空，就会跳过这个 pair：

```python
if not candidates:
    continue
```

### 图片级别统计
- **输入文件中的图片数**: 2,000
- **输出文件中的图片数**: 2,000
- **完全缺失的图片数**: 0
- **存在遗漏 pair 的图片数**: 1,255

### 额外说明
- Stage2 结果中有 26,351 个额外的 pair（在输入文件中不存在），这些是 Stage1 生成的候选 pair，不是输入文件中的 GT pair。

## 建议

### 如果需要提高覆盖率：

1. **对于主要遗漏（4,522 个 pair）**：
   - 需要修改 Stage1 的逻辑，增加 top100 的数量，或者确保所有 GT pair 都进入候选列表
   - 或者修改 Stage2 的逻辑，直接从输入文件中读取所有 GT pair，而不仅仅依赖 Stage1 的结果

2. **对于代码逻辑遗漏（2 个 pair）**：
   - 可以修改 `prepare_data_for_inference` 函数，即使所有候选都是 `"no relation"`，也保留这个 pair 进行处理
   - 或者为 `"no relation"` 的情况也生成 CoT 描述

### 当前状态
- ✅ 所有图片都已被处理（没有完全缺失的图片）
- ⚠️ 约 45.65% 的 GT pair 没有被处理（主要因为 Stage1 中没有这些 pair）
- ⚠️ 2 个 pair 因为代码逻辑被遗漏（可以修复）

## 详细数据文件

遗漏的 pair 详细信息已保存到：
`/public/home/xiaojw2025/Data/stage2/missing_pairs.json`

