# sparse router residual MoE V1 设计稿（2026-04-08）

## 1. 文档目的

本文档记录 `2026-04-08` 确认的 `V1` 方案边界，用于指导：

- `src/try/065_sparse_router_residual_moe_v1/` 的首版实现；
- 后续实验时避免继续把问题重新发散成过大的专家集合或过重的训练流程。

本文档是设计稿，不是实验结论，不写入 `PROJECT.md` 的稳定结论区。

## 2. V1 要回答的问题

`V1` 的首要目标不是追当前全仓库 `SOTA`，而是先回答：

- 统一训练的 `rpm-first + shared encoder + sparse router + bounded residual experts` 是否值得继续；
- 这条 unified 路线是否比当前分拆式 quickcheck 更平滑、更一致；
- 路由行为是否能形成可解释结构，而不是只在个别 split 上偶然刷出更好的 `MAE`。

## 3. V1 明确保留的约束

- `base = rpm_knn4`
- `base` 固定，不参与梯度更新
- shared encoder 复用现有 `TinyTCN 2s+8s` encoder 权重初始化
- 所有 expert 只输出 `bounded residual`
- 显式保留 `no-op expert`
- 保留 `L_noharm`
  - 允许增强
  - 不允许明显伤害 `base`

## 4. V1 明确收缩后的模型边界

### 4.1 experts

`V1` 只保留 `3-expert`：

- `Expert0 = no-op`
- `Expert1 = global residual expert`
- `Expert2 = prototype delta-only residual expert`

当前 `V1` 不引入显式 `domain-shift expert`。

原因：

- `V1` 先验证 unified sparse residual-MoE 本身是否成立；
- 不预先规定某个 expert 天生对应 `added-like` 域；
- 先观察 learned routing 是否会自然把局部域偏移样本送向 prototype residual expert。

### 4.2 router

router 保留一次性 sparse routing 的定义：

- 输入 shared case embedding 与少量 case-level 辅助特征；
- 直接输出 `3` 个 expert logits；
- 使用 sparse `top-k` 激活；
- 不改写成“先开不开、再在残差专家中选 top-1”的两阶段结构。

### 4.3 shared encoder

encoder 继续使用 `TinyTCN 2s+8s` 的共享表征路线：

- `2s encoder`
- `8s encoder`
- case-level pooling
- `embedding_concat`

`V1` 不引入新的大型 backbone，不尝试同时替换 encoder 与 MoE 结构。

## 5. V1 模型定义

## 5.1 预测公式

记：

- `h`：shared encoder 输出的 case embedding
- `alpha_e`：router 对第 `e` 个 expert 的稀疏激活权重
- `delta_e`：第 `e` 个 expert 输出的 bounded residual

则：

```text
base_pred = rpm_knn4(case)
pred = base_pred + sum_e alpha_e * delta_e
```

其中：

- `Expert0` 固定输出 `0`
- 稀疏路由后，仅 `top-k` experts 保留非零权重

## 5.2 Expert0: no-op

- 输入：无
- 输出：`delta_0 = 0`
- 作用：
  - 显式回退到 `base`
  - 为 `final` 侧保护提供默认出口

## 5.3 Expert1: global residual expert

建议输入：

- `h`
- `base_pred`
- `rpm`

建议结构：

- 小型 `MLP`
- hidden dim 先用小规模设置
- 输出经 `tanh * bound` 做残差限幅

该 expert 的目标是：

- 只学习统一的小修正；
- 不承担局部 prototype 检索逻辑。

## 5.4 Expert2: prototype delta-only residual expert

先基于 `h` 在训练池内做 reference retrieval：

- `top-r` case neighbors
- 距离加权求 `prototype h_ref`

建议输入只保留：

- `delta = h - h_ref`
- `|delta|`
- `top1_embed_distance`
- `topk_embed_mean_distance`
- `topk_embed_std_distance`
- `base_pred`

明确不做：

- 不把 `h` 与 `h_ref` 的 full input 一起直接喂给大头；
- 不让 prototype expert 退化成几乎可自由重建最终风速的模块。

该约束直接复用 `2026-04-08` 的 prototype ablation 经验。

## 6. V1 训练流程

`V1` 不采用完整四阶段体系。推荐流程如下：

1. 固定 `base = rpm_knn4`
2. shared encoder 使用现有 `2s+8s` encoder 权重初始化
3. 不单独预训练 `Expert1 / Expert2`
4. 直接训练 `router + experts`
5. 前几 epoch 冻结 encoder，只训练上层
6. 后续以更小学习率解冻 encoder，做联合微调

这样做的目的不是最充分利用全部训练技巧，而是优先让 `V1` 回答方向性问题：

- unified sparse residual-MoE 到底值不值得继续。

## 7. V1 损失函数

`V1` 建议先保留以下核心损失：

### 7.1 主损失

```text
L_main = Huber(pred, y)
```

### 7.2 no-harm 保护损失

```text
L_noharm = max(0, |pred-y| - |base_pred-y| - margin)
```

作用：

- 允许增强；
- 但显式惩罚“比 `base` 更差太多”的预测；
- 尤其保护 `final`。

### 7.3 残差幅度约束

```text
L_delta = mean(sum_e alpha_e * |delta_e|)
```

作用：

- 强化“小修正”角色；
- 防止 expert 抢走主干地位。

### 7.4 总损失

```text
L = L_main + lambda_noharm * L_noharm + lambda_delta * L_delta
```

`V1` 先不强制加入更复杂的 load-balance 项，优先观察实际路由行为是否已经塌缩。

## 8. V1 训练后必须监控的行为

`V1` 的首要观测不是单一 `MAE`，而是路由行为是否合理。

至少要输出并复核：

- `final` 上 `Expert0(no-op)` 的激活比例
- `added-like` 样本上 `Expert2` 的激活比例
- `Expert1` 的平均残差幅度与符号分布
- `Expert2` 的平均残差幅度与检索距离统计
- `worse_than_base_rate`
- `mean_excess_harm`
- 各域上的平均 residual 幅度

若这些行为没有形成结构，即使个别 split 指标略好，也不能视为 `V1` 成立。

## 9. V1 验收标准

`2026-04-08` 版本建议以以下标准验收：

- `final` 不明显差于 `rpm_knn4`
- `added` 至少优于纯 `rpm_knn4`
- 路由行为具有可解释性：
  - `final -> no-op` 更频繁
  - `added-like -> prototype/global residual` 更频繁
- 相比现有 trigger / bucket 路线，统一模型行为更平滑、更一致

## 10. V1 最小 ablation 矩阵

建议最小比较组：

- `A0 = rpm_knn4`
- `A1 = rpm_knn4 + global residual only`
- `A2 = rpm_knn4 + prototype delta-only residual only`
- `A3 = sparse router + 3 experts`
- `A4 = A3 without L_noharm`

这组对照足以回答：

- unified routing 是否优于单 expert；
- prototype expert 是否具有独立价值；
- `L_noharm` 是否真的提供了必要保护。

## 11. 当前不做的事

`V1` 当前明确不做：

- 显式 `domain-shift expert`
- 更大的 expert 集合
- 两阶段 `open / close` router
- 从零训练新 encoder
- 完整四阶段 teacher-style 预训练流程
- 以追全仓库 `SOTA` 作为第一目标

## 12. 对应实现入口

- try 目录：`src/try/065_sparse_router_residual_moe_v1/`
- scaffold 入口：`src/try/065_sparse_router_residual_moe_v1/run_sparse_router_residual_moe_v1.py`

后续若开始编码，应优先保证：

- 目录骨架稳定；
- 输出表结构先固定；
- 指标与行为监控先跑通；
- 再逐步补齐真正训练实现。
