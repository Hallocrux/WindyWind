# 033 去掉难工况后的标签一致性快速检查

## 目标

- 用一个更干净的训练池，检查可疑工况标签是否明显偏离；
- 训练时先去掉当前公认难工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`
- 先用新增且非难工况的 case 做验证：
  - `工况15`
  - `工况16`
  - `工况19`
  - `工况20`
- 如果这一步仍有参考价值，再看可疑工况：
  - `工况1`
  - `工况18`

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- 模型：
  - `TinyTCN@5s`
- 评估方式：
  - 对每个目标工况做 leave-one-case-out
  - 但训练池固定去掉 `1 / 3 / 17 / 18`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/033_label_sanity_check_excluding_hard_cases/run_label_sanity_check_excluding_hard_cases.py
```

## 输出

- 输出目录：`outputs/try/033_label_sanity_check_excluding_hard_cases/`
- 固定产物：
  - `validation_case_predictions.csv`
  - `suspicious_case_predictions.csv`
  - `summary.csv`
  - `summary.md`
