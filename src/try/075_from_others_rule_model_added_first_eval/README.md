# 075 from_others rule model added-first eval

## 目标

- 对 `src/from_others/1/predict_2.py` 做 added-first 口径评测；
- 不改外部来源模型内部逻辑；
- 只包一层统一输入输出与评估。

## 输入

- 外部来源脚本：
  - `src/from_others/1/predict_2.py`
- 数据：
  - `data/added/standardized_datasets/工况21-24.csv`
  - `data/added2/standardized_datasets/工况25-30.csv`
  - `data/added/dataset_manifest.csv`
  - `data/added2/dataset_manifest.csv`

## 评估口径

- 主口径：`added_to_added2`
  - 对规则模型来说不涉及训练；
  - 这里的含义是：
    - 只在 `added2(25-30)` 上汇报主结果；
    - 用于和当前 added-first 主口径对齐。
- 参考口径：`external_loocv`
  - 对规则模型来说同样不涉及拟合；
  - 这里只表示：
    - 在 `added + added2` 的全部带标签工况上统一评估；
    - 作为外部域整体参考。

## 输出

- 输出目录：`outputs/try/075_from_others_rule_model_added_first_eval/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `summary.md`

## 运行方式

```powershell
uv run python src/try/075_from_others_rule_model_added_first_eval/run_from_others_rule_model_added_first_eval.py
```

## 说明

- 本探索不对 `predict_2.py` 做再训练；
- 若后续需要和仓库现有基线同表对照，再在下游脚本里汇总。
