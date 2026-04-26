# 068 domain LODO MoE diagnosis

## 目标

- 对 `067` 的 Kaggle full run 做结构化诊断；
- 回答：
  - 为什么 `domain_0` 上 `A3_sparse_router_moe` 有正信号；
  - 为什么 `domain_2`、`domain_3` 上 `A3_sparse_router_moe` 出现明显退化；
  - `router` 与 `prototype retrieval` 分别在这些域里扮演了什么角色。

## 输入

- `outputs/try/067_reuse_embedding_domain_lodo_moe/latest_kernel_output_v3/windywind/outputs/try/067_reuse_embedding_domain_lodo_moe/`
- `outputs/try/066_reuse_embedding_domain_split/domain_assignment.csv`

## 运行方式

```powershell
uv run python src/try/068_domain_lodo_moe_diagnosis/run_domain_lodo_moe_diagnosis.py
```

## 输出

- 输出目录：`outputs/try/068_domain_lodo_moe_diagnosis/`
- 主要文件：
  - `case_delta_table.csv`
  - `domain_behavior_summary.csv`
  - `router_summary_by_domain.csv`
  - `prototype_neighbor_mix_by_domain.csv`
  - `key_failure_cases.csv`
  - `summary.md`
  - `plots/`

## 说明

- 本探索只做结果诊断，不重新训练模型。
- 默认读取 `latest_kernel_output_v3` 里的远端 full run 输出。
