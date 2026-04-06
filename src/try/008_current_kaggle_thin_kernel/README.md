# 008 current Kaggle thin kernel

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

## 目标

验证 `src/current/` 这条表格主线是否可以在**不修改训练逻辑**的前提下，通过：

- 一个代码 dataset
- 一个数据 dataset
- 一个负责组装目录结构的 thin kernel

在 Kaggle 上跑通一次。

## 输入

- 代码入口：
  - `main.py`
  - `src/__init__.py`
  - `src/current/`
- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`

## 运行方式

先确保本机 `kaggle` CLI 已可用，并且当前账户已经完成认证。

```bash
python src/try/008_current_kaggle_thin_kernel/run_try.py
```

如果 code/data datasets 已经发布过，后续调试 thin kernel 时优先走快路径：

```bash
python src/try/008_current_kaggle_thin_kernel/push_kernel_only.py --username vnvile
```

如需覆盖默认 slug，可显式传参：

```bash
python src/try/008_current_kaggle_thin_kernel/run_try.py ^
  --code-dataset-slug windywind-try-008-current-code ^
  --data-dataset-slug windywind-try-008-current-data ^
  --kernel-slug windywind-try-008-current-thin-kernel
```

## 方法说明

- `2026-04-05` 本探索不直接把整个仓库目录上传为 Kaggle dataset。
- `2026-04-05` 本探索会先把代码与数据分别打成受控 zip payload，再发布到两个独立 dataset。
- `2026-04-05` thin kernel 负责在运行时把两个 payload 解压到 `/kaggle/working/windywind/`，恢复当前代码期望的相对路径结构，然后执行 `main.py`。

## 输出位置

本探索的所有产物统一写到：

- `outputs/try/008_current_kaggle_thin_kernel/`

典型产物包括：

- `runs/<timestamp>/publish_payloads/`
- `runs/<timestamp>/kernel/`
- `runs/<timestamp>/local_smoke/`
- `runs/<timestamp>/remote_outputs/`
- `runs/<timestamp>/summary.json`

## 预期判断口径

- 如果本地 smoke test 通过，说明 payload 结构和 thin kernel 目录组装逻辑成立。
- 如果 Kaggle kernel 成功完成并能回收到 `outputs/*.csv`，说明当前 `src/current/` 可以在**不改训练逻辑**的前提下迁移到 Kaggle。
- 如果失败，优先判断：
  - packaging / 路径问题
  - Kaggle 运行环境问题
  - 代码本身问题
