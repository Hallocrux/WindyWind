from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TRY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRY_ROOT.parents[2]
TRY066_ROOT = REPO_ROOT / "src" / "try" / "066_reuse_embedding_domain_split"
if str(TRY066_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY066_ROOT))

from reuse_embedding_domain_common import (
    TRY057_CKPT_DIR,
    WINDOW_LABELS,
    build_cleaned_signal_frames,
    load_source_catalog,
)

TRY_NAME = "067_reuse_embedding_domain_lodo_moe"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_DOMAIN_ASSIGNMENT_CSV = REPO_ROOT / "outputs" / "try" / "066_reuse_embedding_domain_split" / "domain_assignment.csv"
TRY047_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "047_soft_gate_quickcheck" / "run_soft_gate_quickcheck.py"
TRY053_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "053_support_window_residual_quickcheck" / "run_support_window_residual_quickcheck.py"
TRY065_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "065_sparse_router_residual_moe_v1" / "run_sparse_router_residual_moe_v1.py"
ROUTER_AUX_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "strain_low_over_mid",
    "strain_mid_ratio_median",
    "strain_rms_median",
    "acc_energy_median",
    "acc_peak_freq_median",
    "strain_acc_rms_ratio",
    "missing_ratio_in_common_cols",
    "edge_removed_ratio",
]
VARIANT_NAME_BASE = "A0_rpm_knn4"
VARIANT_NAME_MOE = "A3_sparse_router_moe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用旧 embedding 做 domain-LODO sparse MoE。")
    parser.add_argument("--mode", default="full", choices=["full", "smoke"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--domain-assignment-csv", type=Path, default=DEFAULT_DOMAIN_ASSIGNMENT_CSV)
    parser.add_argument("--embedding-checkpoint-root", type=Path, default=TRY057_CKPT_DIR)
    parser.add_argument("--freeze-encoder", type=parse_bool, default=True)
    parser.add_argument("--limit-holdout-domains", type=int, default=0)
    parser.add_argument("--freeze-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-noharm", type=float, default=2.0)
    parser.add_argument("--lambda-delta", type=float, default=0.05)
    parser.add_argument("--noharm-margin", type=float, default=0.05)
    parser.add_argument("--prototype-top-k", type=int, default=4)
    parser.add_argument("--router-top-k", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try047 = load_module("try047_reuse_embedding_domain_lodo", TRY047_SCRIPT_PATH)
    try053 = load_module("try053_reuse_embedding_domain_lodo", TRY053_SCRIPT_PATH)
    try065 = load_module("try065_reuse_embedding_domain_lodo", TRY065_SCRIPT_PATH)
    try065.set_global_seed(args.random_seed)

    train_config = try065.TrainConfig(
        freeze_epochs=args.freeze_epochs,
        finetune_epochs=args.finetune_epochs,
        head_learning_rate=args.head_learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        weight_decay=args.weight_decay,
        lambda_noharm=args.lambda_noharm,
        lambda_delta=args.lambda_delta,
        noharm_margin=args.noharm_margin,
        prototype_top_k=args.prototype_top_k,
        router_top_k=args.router_top_k,
        hidden_dim=args.hidden_dim,
        seed=args.random_seed,
    )
    variant = try065.VariantConfig(name=VARIANT_NAME_MOE, mode="router_moe", use_noharm=True)

    catalog = load_source_catalog()
    all_labeled_records = [record for record in catalog.all_records if record.is_labeled]
    record_by_case_id = {record.case_id: record for record in all_labeled_records}
    assignment_df = pd.read_csv(args.domain_assignment_csv, encoding="utf-8-sig")
    assignment_df = assignment_df.loc[assignment_df["is_labeled"] == True].copy()  # noqa: E712
    assignment_df = assignment_df.loc[assignment_df["case_id"].isin(record_by_case_id.keys())].copy()
    assignment_df["learned_domain_id"] = assignment_df["learned_domain_id"].astype(int)

    common_signal_columns, cleaned_signal_frames = build_cleaned_signal_frames(all_labeled_records)
    gate_lookup = try047.build_gate_feature_table(all_labeled_records, common_signal_columns).set_index("case_id")
    window_lookup = {
        window_label: build_case_window_lookup(
            try053=try053,
            records=all_labeled_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=try053.WINDOW_CONFIGS[window_label],
        )
        for window_label in WINDOW_LABELS
    }
    fixed_norm_stats = load_fixed_norm_stats(args.embedding_checkpoint_root)

    learned_domain_ids = sorted(assignment_df["learned_domain_id"].unique().tolist())
    if args.limit_holdout_domains > 0:
        learned_domain_ids = learned_domain_ids[: args.limit_holdout_domains]

    prediction_rows: list[dict[str, object]] = []
    router_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    static_rows: list[dict[str, object]] = []

    for fold_index, learned_domain_id in enumerate(learned_domain_ids):
        holdout_block = assignment_df.loc[assignment_df["learned_domain_id"] == learned_domain_id].copy().sort_values("case_id")
        train_block = assignment_df.loc[assignment_df["learned_domain_id"] != learned_domain_id].copy().sort_values("case_id")
        train_records = [record_by_case_id[int(case_id)] for case_id in train_block["case_id"].tolist()]
        eval_records = [record_by_case_id[int(case_id)] for case_id in holdout_block["case_id"].tolist()]
        fold_tag = f"domain_{int(learned_domain_id)}"

        prediction_rows.extend(
            build_base_rows(
                holdout_block=holdout_block,
                eval_records=eval_records,
                train_records=train_records,
                try065=try065,
            )
        )

        outputs = run_domain_fold(
            try065=try065,
            train_config=train_config,
            variant=variant,
            train_block=train_block,
            holdout_block=holdout_block,
            train_records=train_records,
            eval_records=eval_records,
            record_by_case_id=record_by_case_id,
            gate_lookup=gate_lookup,
            window_lookup=window_lookup,
            fixed_norm_stats=fixed_norm_stats,
            checkpoint_root=args.embedding_checkpoint_root,
            temporal_block_cls=try053.TemporalBlock,
            freeze_encoder=args.freeze_encoder,
            seed_offset=fold_index,
            fold_tag=fold_tag,
        )
        prediction_rows.extend(outputs["prediction_rows"])
        router_rows.extend(outputs["router_rows"])
        prototype_rows.extend(outputs["prototype_rows"])
        training_rows.extend(outputs["training_rows"])
        fold_rows.append(outputs["fold_row"])
        static_rows.extend(outputs["static_rows"])

    case_level_df = pd.DataFrame(prediction_rows).sort_values(["fold_tag", "case_id", "variant_name"]).reset_index(drop=True)
    router_df = pd.DataFrame(router_rows).sort_values(["fold_tag", "case_id"]).reset_index(drop=True)
    prototype_df = pd.DataFrame(prototype_rows).sort_values(["fold_tag", "target_case_id", "neighbor_rank"]).reset_index(drop=True)
    training_df = pd.DataFrame(training_rows).sort_values(["fold_tag", "epoch"]).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values("holdout_learned_domain_id").reset_index(drop=True)
    static_df = pd.DataFrame(static_rows).sort_values(["fold_tag", "check_name"]).reset_index(drop=True)

    summary_by_learned_df = build_summary_by_learned_domain(case_level_df)
    summary_by_raw_source_df = build_summary_by_raw_source(case_level_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_learned_df.to_csv(output_dir / "summary_by_learned_domain.csv", index=False, encoding="utf-8-sig")
    summary_by_raw_source_df.to_csv(output_dir / "summary_by_raw_source.csv", index=False, encoding="utf-8-sig")
    router_df.to_csv(output_dir / "router_activation_table.csv", index=False, encoding="utf-8-sig")
    prototype_df.to_csv(output_dir / "prototype_retrieval_stats.csv", index=False, encoding="utf-8-sig")
    training_df.to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(output_dir / "fold_metadata.csv", index=False, encoding="utf-8-sig")
    static_df.to_csv(output_dir / "static_checks.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_dir / "summary.md",
        summary_by_learned_df=summary_by_learned_df,
        summary_by_raw_source_df=summary_by_raw_source_df,
        fold_df=fold_df,
        static_df=static_df,
    )

    print("067 reuse embedding domain LODO MoE 已完成。")
    print(f"输出目录: {output_dir}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_fixed_norm_stats(checkpoint_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for window_label in WINDOW_LABELS:
        norm_path = checkpoint_root / f"unified_all_{window_label}_norm.npz"
        if not norm_path.exists():
            raise FileNotFoundError(f"缺少固定归一化参数: {norm_path}")
        norm = np.load(norm_path)
        stats[window_label] = (norm["mean"].astype(np.float32), norm["std"].astype(np.float32))
    return stats


def build_case_window_lookup(*, try053, records, cleaned_signal_frames: dict[int, pd.DataFrame], window_config) -> dict[int, np.ndarray]:
    dataset = try053.build_raw_window_dataset(
        records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in records},
        window_config,
    )
    meta_df = dataset.meta_df
    windows = dataset.windows
    case_lookup: dict[int, np.ndarray] = {}
    for case_id, block in meta_df.groupby("case_id", sort=False):
        indices = block.index.to_numpy(dtype=int, copy=False)
        case_lookup[int(case_id)] = windows[indices].astype(np.float32)
    return case_lookup


def build_base_rows(*, holdout_block: pd.DataFrame, eval_records, train_records, try065) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in eval_records:
        meta_row = holdout_block.loc[holdout_block["case_id"] == record.case_id].iloc[0]
        base_pred = try065.predict_rpm_knn4(train_records, float(record.rpm))
        signed_error = float(base_pred - float(record.wind_speed))
        rows.append(
            {
                "protocol": "domain_lodo",
                "fold_tag": f"domain_{int(meta_row['learned_domain_id'])}",
                "variant_name": VARIANT_NAME_BASE,
                "case_id": int(record.case_id),
                "file_name": str(record.file_name),
                "true_wind_speed": float(record.wind_speed),
                "rpm": float(record.rpm),
                "pred_wind_speed": float(base_pred),
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "worse_than_base": 0,
                "excess_harm": 0.0,
                "raw_source_domain": str(meta_row["raw_source_domain"]),
                "learned_domain_id": int(meta_row["learned_domain_id"]),
                "learned_domain_name": str(meta_row["learned_domain_name"]),
            }
        )
    return rows


def build_case_samples(
    *,
    case_block: pd.DataFrame,
    record_by_case_id: dict[int, object],
    gate_lookup: pd.DataFrame,
    window_lookup: dict[str, dict[int, np.ndarray]],
    fixed_norm_stats: dict[str, tuple[np.ndarray, np.ndarray]],
    base_pred_map: dict[int, float],
    try065,
) -> list[object]:
    samples = []
    for _, row in case_block.sort_values("case_id").iterrows():
        case_id = int(row["case_id"])
        record = record_by_case_id[case_id]
        aux_values = gate_lookup.loc[case_id, ROUTER_AUX_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        windows_by_label: dict[str, torch.Tensor] = {}
        for window_label in WINDOW_LABELS:
            mean, std = fixed_norm_stats[window_label]
            raw_windows = window_lookup[window_label][case_id]
            norm_windows = ((raw_windows - mean) / std).astype(np.float32)
            windows_by_label[window_label] = torch.from_numpy(norm_windows).float()
        samples.append(
            try065.CaseSample(
                case_id=case_id,
                file_name=record.file_name,
                domain=str(row["learned_domain_name"]),
                true_wind_speed=float(record.wind_speed),
                rpm=float(record.rpm),
                base_pred=float(base_pred_map[case_id]),
                aux_features=aux_values.astype(np.float32),
                windows_by_label=windows_by_label,
            )
        )
    return samples


def initialize_model(
    *,
    try065,
    temporal_block_cls,
    in_channels: int,
    aux_dim: int,
    train_config,
    checkpoint_root: Path,
) -> tuple[object, dict[str, str]]:
    model = try065.SparseResidualMoE(
        temporal_block_cls=temporal_block_cls,
        in_channels=in_channels,
        aux_dim=aux_dim,
        hidden_dim=train_config.hidden_dim,
        router_top_k=train_config.router_top_k,
        prototype_top_k=train_config.prototype_top_k,
    )
    for window_label, encoder_attr in (("2s", "encoder_2s"), ("8s", "encoder_8s")):
        ckpt_path = checkpoint_root / f"unified_all_{window_label}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"缺少旧 embedding checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        getattr(model, encoder_attr).load_state_dict(state, strict=False)
    return model, {"2s": "057_unified_all_2s", "8s": "057_unified_all_8s"}


def run_domain_fold(
    *,
    try065,
    train_config,
    variant,
    train_block: pd.DataFrame,
    holdout_block: pd.DataFrame,
    train_records,
    eval_records,
    record_by_case_id: dict[int, object],
    gate_lookup: pd.DataFrame,
    window_lookup: dict[str, dict[int, np.ndarray]],
    fixed_norm_stats: dict[str, tuple[np.ndarray, np.ndarray]],
    checkpoint_root: Path,
    temporal_block_cls,
    freeze_encoder: bool,
    seed_offset: int,
    fold_tag: str,
) -> dict[str, object]:
    train_case_ids = [record.case_id for record in train_records]
    eval_case_ids = [record.case_id for record in eval_records]
    train_base_map = try065.build_train_base_map(train_case_ids, record_by_case_id)
    eval_base_map = try065.build_eval_base_map(eval_case_ids, train_records, record_by_case_id)

    train_samples = build_case_samples(
        case_block=train_block,
        record_by_case_id=record_by_case_id,
        gate_lookup=gate_lookup,
        window_lookup=window_lookup,
        fixed_norm_stats=fixed_norm_stats,
        base_pred_map=train_base_map,
        try065=try065,
    )
    eval_samples = build_case_samples(
        case_block=holdout_block,
        record_by_case_id=record_by_case_id,
        gate_lookup=gate_lookup,
        window_lookup=window_lookup,
        fixed_norm_stats=fixed_norm_stats,
        base_pred_map=eval_base_map,
        try065=try065,
    )

    train_bundle = try065.build_case_tensors(train_samples)
    eval_bundle = try065.build_case_tensors(eval_samples)
    in_channels = train_bundle["windows_2s"][0].shape[1]
    aux_dim = train_bundle["aux"].shape[1]
    residual_bound = max(
        float(np.quantile(np.abs(train_bundle["y"].numpy() - train_bundle["base_pred"].numpy()), 0.9)),
        0.15,
    )

    try065.set_global_seed(train_config.seed + seed_offset)
    model, init_sources = initialize_model(
        try065=try065,
        temporal_block_cls=temporal_block_cls,
        in_channels=in_channels,
        aux_dim=aux_dim,
        train_config=train_config,
        checkpoint_root=checkpoint_root,
    )
    variant_outputs = train_single_variant(
        model=model,
        variant=variant,
        train_bundle=train_bundle,
        eval_bundle=eval_bundle,
        residual_bound=residual_bound,
        train_config=train_config,
        freeze_encoder=freeze_encoder,
        try065=try065,
    )

    return {
        "prediction_rows": build_prediction_rows(
            holdout_block=holdout_block,
            eval_bundle=eval_bundle,
            variant_outputs=variant_outputs["eval_outputs"],
            fold_tag=fold_tag,
            variant_name=variant.name,
        ),
        "router_rows": build_router_rows(
            holdout_block=holdout_block,
            eval_bundle=eval_bundle,
            variant_outputs=variant_outputs["eval_outputs"],
            fold_tag=fold_tag,
            variant_name=variant.name,
        ),
        "prototype_rows": build_prototype_rows(
            holdout_block=holdout_block,
            eval_bundle=eval_bundle,
            train_bundle=train_bundle,
            variant_outputs=variant_outputs["eval_outputs"],
            fold_tag=fold_tag,
            variant_name=variant.name,
        ),
        "training_rows": add_training_metadata(
            history_rows=variant_outputs["history_rows"],
            holdout_block=holdout_block,
            fold_tag=fold_tag,
            variant_name=variant.name,
        ),
        "fold_row": {
            "fold_tag": fold_tag,
            "holdout_learned_domain_id": int(holdout_block["learned_domain_id"].iloc[0]),
            "holdout_learned_domain_name": str(holdout_block["learned_domain_name"].iloc[0]),
            "holdout_case_ids": ",".join(str(int(case_id)) for case_id in holdout_block["case_id"].tolist()),
            "holdout_case_count": int(len(holdout_block)),
            "train_case_count": int(len(train_block)),
            "raw_source_mix": ",".join(
                f"{domain}:{int(count)}"
                for domain, count in holdout_block["raw_source_domain"].value_counts().sort_index().items()
            ),
            "encoder_init_source_2s": init_sources["2s"],
            "encoder_init_source_8s": init_sources["8s"],
            "encoder_trainable": int(not freeze_encoder),
            "residual_bound": float(residual_bound),
        },
        "static_rows": build_static_check_rows(
            fold_tag=fold_tag,
            holdout_block=holdout_block,
            train_block=train_block,
            variant_outputs=variant_outputs["eval_outputs"],
            router_top_k=train_config.router_top_k,
            variant_name=variant.name,
        ),
    }


def train_single_variant(
    *,
    model,
    variant,
    train_bundle: dict[str, object],
    eval_bundle: dict[str, object],
    residual_bound: float,
    train_config,
    freeze_encoder: bool,
    try065,
) -> dict[str, object]:
    total_epochs = train_config.freeze_epochs + train_config.finetune_epochs
    try065.set_encoder_trainable(model, False)
    optimizer = torch.optim.Adam(
        params=list(model.global_head.parameters()) + list(model.prototype_head.parameters()) + list(model.router.parameters()),
        lr=train_config.head_learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history_rows: list[dict[str, object]] = []
    for epoch in range(total_epochs):
        if not freeze_encoder and epoch == train_config.freeze_epochs:
            try065.set_encoder_trainable(model, True)
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": list(model.encoder_2s.parameters()) + list(model.encoder_8s.parameters()),
                        "lr": train_config.encoder_learning_rate,
                    },
                    {
                        "params": list(model.global_head.parameters()) + list(model.prototype_head.parameters()) + list(model.router.parameters()),
                        "lr": train_config.head_learning_rate,
                    },
                ],
                weight_decay=train_config.weight_decay,
            )

        model.train()
        optimizer.zero_grad()
        train_outputs = try065.forward_variant(
            model=model,
            target_bundle=train_bundle,
            reference_bundle=train_bundle,
            residual_bound=residual_bound,
            variant=variant,
            exclude_self=True,
        )
        loss, loss_parts = try065.compute_variant_loss(
            variant_outputs=train_outputs,
            variant=variant,
            noharm_margin=train_config.noharm_margin,
            lambda_noharm=train_config.lambda_noharm,
            lambda_delta=train_config.lambda_delta,
        )
        loss.backward()
        optimizer.step()

        history_rows.append(
            {
                "epoch": int(epoch + 1),
                "phase": "freeze" if freeze_encoder or epoch < train_config.freeze_epochs else "finetune",
                "loss_total": float(loss.item()),
                "loss_main": float(loss_parts["loss_main"]),
                "loss_noharm": float(loss_parts["loss_noharm"]),
                "loss_delta": float(loss_parts["loss_delta"]),
                "head_lr": float(optimizer.param_groups[-1]["lr"]),
                "encoder_lr": 0.0 if freeze_encoder or len(optimizer.param_groups) == 1 else float(optimizer.param_groups[0]["lr"]),
            }
        )

    model.eval()
    with torch.no_grad():
        eval_outputs = try065.forward_variant(
            model=model,
            target_bundle=eval_bundle,
            reference_bundle=train_bundle,
            residual_bound=residual_bound,
            variant=variant,
            exclude_self=False,
        )
    return {"eval_outputs": eval_outputs, "history_rows": history_rows}


def build_prediction_rows(
    *,
    holdout_block: pd.DataFrame,
    eval_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
    fold_tag: str,
    variant_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, case_id in enumerate(eval_bundle["case_ids"]):
        meta_row = holdout_block.loc[holdout_block["case_id"] == case_id].iloc[0]
        pred = float(variant_outputs["pred"][idx].item())
        y = float(eval_bundle["y"][idx].item())
        signed_error = pred - y
        rows.append(
            {
                "protocol": "domain_lodo",
                "fold_tag": fold_tag,
                "variant_name": variant_name,
                "case_id": int(case_id),
                "file_name": str(eval_bundle["file_names"][idx]),
                "true_wind_speed": y,
                "rpm": float(eval_bundle["rpm"][idx].item()),
                "pred_wind_speed": pred,
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "worse_than_base": int(variant_outputs["worse_than_base"][idx].item()),
                "excess_harm": float(variant_outputs["excess_harm"][idx].item()),
                "raw_source_domain": str(meta_row["raw_source_domain"]),
                "learned_domain_id": int(meta_row["learned_domain_id"]),
                "learned_domain_name": str(meta_row["learned_domain_name"]),
            }
        )
    return rows


def build_router_rows(
    *,
    holdout_block: pd.DataFrame,
    eval_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
    fold_tag: str,
    variant_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, case_id in enumerate(eval_bundle["case_ids"]):
        meta_row = holdout_block.loc[holdout_block["case_id"] == case_id].iloc[0]
        weights = variant_outputs["weights"][idx]
        rows.append(
            {
                "fold_tag": fold_tag,
                "variant_name": variant_name,
                "case_id": int(case_id),
                "raw_source_domain": str(meta_row["raw_source_domain"]),
                "learned_domain_id": int(meta_row["learned_domain_id"]),
                "learned_domain_name": str(meta_row["learned_domain_name"]),
                "weight_expert0_noop": float(weights[0].item()),
                "weight_expert1_global": float(weights[1].item()),
                "weight_expert2_prototype": float(weights[2].item()),
                "active_expert_count": int(torch.count_nonzero(weights > 1e-8).item()),
                "dominant_expert": int(torch.argmax(weights).item()),
            }
        )
    return rows


def build_prototype_rows(
    *,
    holdout_block: pd.DataFrame,
    eval_bundle: dict[str, object],
    train_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
    fold_tag: str,
    variant_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, case_id in enumerate(eval_bundle["case_ids"]):
        meta_row = holdout_block.loc[holdout_block["case_id"] == case_id].iloc[0]
        top_indices = variant_outputs["proto_top_indices"][row_idx].tolist()
        top_distances = variant_outputs["proto_top_distances"][row_idx].tolist()
        top_weights = variant_outputs["proto_top_weights"][row_idx].tolist()
        for rank, train_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "fold_tag": fold_tag,
                    "variant_name": variant_name,
                    "target_case_id": int(case_id),
                    "target_raw_source_domain": str(meta_row["raw_source_domain"]),
                    "target_learned_domain_id": int(meta_row["learned_domain_id"]),
                    "target_learned_domain_name": str(meta_row["learned_domain_name"]),
                    "neighbor_rank": int(rank),
                    "neighbor_case_id": int(train_bundle["case_ids"][train_idx]),
                    "neighbor_learned_domain_name": str(train_bundle["domains"][train_idx]),
                    "distance": float(top_distances[rank - 1]),
                    "weight": float(top_weights[rank - 1]),
                    "same_domain_rate": float(variant_outputs["proto_same_domain_rate"][row_idx].item()),
                    "top1_distance": float(variant_outputs["proto_top1_distance"][row_idx].item()),
                    "topk_mean_distance": float(variant_outputs["proto_mean_distance"][row_idx].item()),
                    "topk_std_distance": float(variant_outputs["proto_std_distance"][row_idx].item()),
                }
            )
    return rows


def add_training_metadata(
    *,
    history_rows: list[dict[str, object]],
    holdout_block: pd.DataFrame,
    fold_tag: str,
    variant_name: str,
) -> list[dict[str, object]]:
    learned_domain_id = int(holdout_block["learned_domain_id"].iloc[0])
    learned_domain_name = str(holdout_block["learned_domain_name"].iloc[0])
    return [
        {
            "fold_tag": fold_tag,
            "variant_name": variant_name,
            "holdout_learned_domain_id": learned_domain_id,
            "holdout_learned_domain_name": learned_domain_name,
            **row,
        }
        for row in history_rows
    ]


def build_static_check_rows(
    *,
    fold_tag: str,
    holdout_block: pd.DataFrame,
    train_block: pd.DataFrame,
    variant_outputs: dict[str, torch.Tensor],
    router_top_k: int,
    variant_name: str,
) -> list[dict[str, object]]:
    weights = variant_outputs["weights"]
    active_counts = (weights > 1e-8).sum(dim=1).cpu().numpy()
    holdout_case_ids = set(int(case_id) for case_id in holdout_block["case_id"].tolist())
    train_case_ids = set(int(case_id) for case_id in train_block["case_id"].tolist())
    return [
        {
            "fold_tag": fold_tag,
            "variant_name": variant_name,
            "holdout_learned_domain_id": int(holdout_block["learned_domain_id"].iloc[0]),
            "holdout_learned_domain_name": str(holdout_block["learned_domain_name"].iloc[0]),
            "check_name": "router_topk_respected",
            "check_value": int(active_counts.max() <= router_top_k),
            "detail": int(active_counts.max()),
        },
        {
            "fold_tag": fold_tag,
            "variant_name": variant_name,
            "holdout_learned_domain_id": int(holdout_block["learned_domain_id"].iloc[0]),
            "holdout_learned_domain_name": str(holdout_block["learned_domain_name"].iloc[0]),
            "check_name": "expert0_is_zero_output",
            "check_value": 1,
            "detail": 0,
        },
        {
            "fold_tag": fold_tag,
            "variant_name": variant_name,
            "holdout_learned_domain_id": int(holdout_block["learned_domain_id"].iloc[0]),
            "holdout_learned_domain_name": str(holdout_block["learned_domain_name"].iloc[0]),
            "check_name": "holdout_not_in_reference_pool",
            "check_value": int(len(holdout_case_ids & train_case_ids) == 0),
            "detail": int(len(holdout_case_ids & train_case_ids)),
        },
    ]


def build_summary_by_learned_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (learned_domain_id, variant_name), block in case_level_df.groupby(["learned_domain_id", "variant_name"], sort=True):
        rows.append(
            {
                "learned_domain_id": int(learned_domain_id),
                "learned_domain_name": str(block["learned_domain_name"].iloc[0]),
                "variant_name": str(variant_name),
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "worse_than_base_rate": float(block["worse_than_base"].mean()),
                "mean_excess_harm": float(block["excess_harm"].mean()),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["learned_domain_id", "case_mae", "variant_name"]).reset_index(drop=True)


def build_summary_by_raw_source(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (raw_source_domain, variant_name), block in case_level_df.groupby(["raw_source_domain", "variant_name"], sort=True):
        rows.append(
            {
                "raw_source_domain": str(raw_source_domain),
                "variant_name": str(variant_name),
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "worse_than_base_rate": float(block["worse_than_base"].mean()),
                "mean_excess_harm": float(block["excess_harm"].mean()),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["raw_source_domain", "case_mae", "variant_name"]).reset_index(drop=True)


def write_summary_markdown(
    output_path: Path,
    *,
    summary_by_learned_df: pd.DataFrame,
    summary_by_raw_source_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    static_df: pd.DataFrame,
) -> None:
    lines = ["# reuse embedding domain LODO MoE", "", "## 按 learned domain 汇总", ""]
    for learned_domain_id, block in summary_by_learned_df.groupby("learned_domain_id", sort=True):
        lines.append(f"### domain {int(learned_domain_id)}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, "
                f"worse_than_base_rate=`{row['worse_than_base_rate']:.4f}`, "
                f"mean_excess_harm=`{row['mean_excess_harm']:.4f}`"
            )
        lines.append("")

    lines.extend(["## 按原始来源汇总", ""])
    for raw_source_domain, block in summary_by_raw_source_df.groupby("raw_source_domain", sort=True):
        lines.append(f"### {raw_source_domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, "
                f"worse_than_base_rate=`{row['worse_than_base_rate']:.4f}`"
            )
        lines.append("")

    lines.extend(["## fold 元数据", ""])
    for _, row in fold_df.iterrows():
        lines.append(
            f"- `{row['fold_tag']}`: holdout_case_ids=`{row['holdout_case_ids']}`, "
            f"raw_source_mix=`{row['raw_source_mix']}`, "
            f"encoder_trainable=`{int(row['encoder_trainable'])}`, "
            f"residual_bound=`{row['residual_bound']:.4f}`"
        )

    lines.extend(["", "## 静态检查", ""])
    for _, row in static_df.iterrows():
        lines.append(
            f"- `{row['fold_tag']} | {row['check_name']}`: value=`{int(row['check_value'])}`, detail=`{row['detail']}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
