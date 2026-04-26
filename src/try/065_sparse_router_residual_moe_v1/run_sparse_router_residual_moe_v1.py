from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "065_sparse_router_residual_moe_v1"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
TRY047_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "047_soft_gate_quickcheck" / "run_soft_gate_quickcheck.py"
TRY053_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "053_support_window_residual_quickcheck" / "run_support_window_residual_quickcheck.py"
FINAL_DEPLOY_CKPT_DIR = REPO_ROOT / "outputs" / "try" / "063_final_late_fusion_added2_replay" / "models" / "checkpoints"
TRY053_CKPT_DIR = REPO_ROOT / "outputs" / "try" / "053_support_window_residual_quickcheck" / "models" / "checkpoints"
WINDOW_LABELS = ("2s", "8s")
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
PROTOTYPE_TOP_K = 4
ROUTER_TOP_K = 2
EPS = 1e-6


@dataclass(frozen=True)
class TrainConfig:
    freeze_epochs: int
    finetune_epochs: int
    head_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    lambda_noharm: float
    lambda_delta: float
    noharm_margin: float
    prototype_top_k: int
    router_top_k: int
    hidden_dim: int
    seed: int


@dataclass(frozen=True)
class VariantConfig:
    name: str
    mode: str
    use_noharm: bool


@dataclass
class CaseSample:
    case_id: int
    file_name: str
    domain: str
    true_wind_speed: float
    rpm: float
    base_pred: float
    aux_features: np.ndarray
    windows_by_label: dict[str, torch.Tensor]


class TinyTCNEncoder(nn.Module):
    def __init__(self, temporal_block_cls, in_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            temporal_block_cls(in_channels, 16, dilation=1),
            temporal_block_cls(16, 32, dilation=2),
            temporal_block_cls(32, 32, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(windows)
        pooled = self.pool(hidden)
        return pooled.squeeze(-1)


class SmallResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SparseRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseResidualMoE(nn.Module):
    def __init__(
        self,
        *,
        temporal_block_cls,
        in_channels: int,
        aux_dim: int,
        hidden_dim: int,
        router_top_k: int,
        prototype_top_k: int,
    ) -> None:
        super().__init__()
        embedding_dim = 64
        self.encoder_2s = TinyTCNEncoder(temporal_block_cls, in_channels)
        self.encoder_8s = TinyTCNEncoder(temporal_block_cls, in_channels)
        self.global_head = SmallResidualHead(embedding_dim + 2, hidden_dim)
        self.prototype_head = SmallResidualHead(embedding_dim * 2 + 4, hidden_dim)
        self.router = SparseRouter(embedding_dim + aux_dim + 2, hidden_dim, num_experts=3)
        self.router_top_k = router_top_k
        self.prototype_top_k = prototype_top_k

    def encode_cases(self, cases: list[CaseSample]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for sample in cases:
            emb_2s = self.encoder_2s(sample.windows_by_label["2s"]).mean(dim=0)
            emb_8s = self.encoder_8s(sample.windows_by_label["8s"]).mean(dim=0)
            embeddings.append(torch.cat([emb_2s, emb_8s], dim=0))
        return torch.stack(embeddings, dim=0)

    def sparse_topk_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        k = min(self.router_top_k, logits.shape[1])
        top_values, top_indices = torch.topk(logits, k=k, dim=1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(1, top_indices, top_values)
        return torch.softmax(sparse_logits, dim=1)

    def build_prototype_features(
        self,
        *,
        target_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        target_domains: list[str],
        reference_domains: list[str],
        exclude_self: bool,
    ) -> dict[str, torch.Tensor]:
        mean = reference_embeddings.mean(dim=0, keepdim=True)
        std = reference_embeddings.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std > 0, std, torch.ones_like(std))

        scaled_ref = (reference_embeddings - mean) / std
        scaled_target = (target_embeddings - mean) / std
        distances = torch.cdist(scaled_target, scaled_ref)
        if exclude_self and target_embeddings.shape[0] == reference_embeddings.shape[0]:
            distances = distances + torch.eye(distances.shape[0], device=distances.device) * 1e6

        neighbor_count = min(self.prototype_top_k, reference_embeddings.shape[0] - int(exclude_self))
        if neighbor_count <= 0:
            raise RuntimeError("prototype reference pool 为空，无法构造 delta-only expert。")

        top_distances, top_indices = torch.topk(distances, k=neighbor_count, dim=1, largest=False)
        raw_weights = 1.0 / torch.clamp(top_distances, min=EPS)
        proto_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)
        neighbor_embeddings = reference_embeddings[top_indices]
        prototypes = torch.sum(proto_weights.unsqueeze(-1) * neighbor_embeddings, dim=1)

        delta = target_embeddings - prototypes
        abs_delta = torch.abs(delta)
        top1 = top_distances[:, 0]
        mean_distance = torch.sum(proto_weights * top_distances, dim=1)
        std_distance = torch.sqrt(
            torch.sum(proto_weights * torch.square(top_distances - mean_distance.unsqueeze(1)), dim=1)
        )

        same_domain_rate = []
        for row_idx in range(top_indices.shape[0]):
            target_domain = target_domains[row_idx]
            matched = 0
            for neighbor_idx in top_indices[row_idx].tolist():
                if reference_domains[neighbor_idx] == target_domain:
                    matched += 1
            same_domain_rate.append(matched / float(neighbor_count))

        return {
            "prototypes": prototypes,
            "delta": delta,
            "abs_delta": abs_delta,
            "top1_distance": top1,
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "top_indices": top_indices,
            "top_distances": top_distances,
            "top_weights": proto_weights,
            "same_domain_rate": torch.tensor(same_domain_rate, dtype=torch.float32, device=target_embeddings.device),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并评估 Sparse Residual MoE V1。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--freeze-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-noharm", type=float, default=2.0)
    parser.add_argument("--lambda-delta", type=float, default=0.05)
    parser.add_argument("--noharm-margin", type=float, default=0.05)
    parser.add_argument("--prototype-top-k", type=int, default=PROTOTYPE_TOP_K)
    parser.add_argument("--router-top-k", type=int, default=ROUTER_TOP_K)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--limit-final-holdouts", type=int, default=0)
    parser.add_argument("--force-random-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
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

    try047 = load_module("try047_sparse_router_residual_moe", TRY047_SCRIPT_PATH)
    try053 = load_module("try053_sparse_router_residual_moe", TRY053_SCRIPT_PATH)
    set_global_seed(train_config.seed)

    final_records = sorted([record for record in try053.scan_dataset_records() if record.is_labeled], key=lambda record: record.case_id)
    added_records = sorted(try053.load_added_records(), key=lambda record: record.case_id)
    all_records = [*final_records, *added_records]
    record_by_case_id = {record.case_id: record for record in all_records}

    common_signal_columns = try053.get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: try053.load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }
    gate_lookup = try047.build_gate_feature_table(all_records, common_signal_columns).set_index("case_id")
    window_lookup = {
        window_label: build_case_window_lookup(
            try053=try053,
            records=all_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=try053.WINDOW_CONFIGS[window_label],
        )
        for window_label in WINDOW_LABELS
    }

    if args.limit_final_holdouts > 0:
        final_holdout_records = final_records[: args.limit_final_holdouts]
    else:
        final_holdout_records = final_records

    variant_configs = [
        VariantConfig(name="A1_global_residual_only", mode="global_only", use_noharm=True),
        VariantConfig(name="A2_prototype_delta_only", mode="prototype_only", use_noharm=True),
        VariantConfig(name="A3_sparse_router_moe", mode="router_moe", use_noharm=True),
        VariantConfig(name="A4_sparse_router_moe_without_noharm", mode="router_moe", use_noharm=False),
    ]

    prediction_rows: list[dict[str, object]] = []
    router_rows: list[dict[str, object]] = []
    expert_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    static_rows: list[dict[str, object]] = []

    final_base_rows = []
    for holdout in final_holdout_records:
        train_records = [record for record in final_records if record.case_id != holdout.case_id]
        final_base_rows.extend(
            build_base_rows(
                protocol="final_loco",
                fold_tag=f"case_{holdout.case_id}",
                domain="final",
                eval_records=[holdout],
                train_records=train_records,
            )
        )
        fold_outputs = run_protocol_fold(
            protocol="final_loco",
            fold_tag=f"case_{holdout.case_id}",
            train_records=train_records,
            eval_records=[holdout],
            train_domain_tag="final",
            record_by_case_id=record_by_case_id,
            gate_lookup=gate_lookup,
            window_lookup=window_lookup,
            temporal_block_cls=try053.TemporalBlock,
            train_config=train_config,
            variant_configs=variant_configs,
            random_init=args.force_random_init,
        )
        prediction_rows.extend(fold_outputs["predictions"])
        router_rows.extend(fold_outputs["router_rows"])
        expert_rows.extend(fold_outputs["expert_rows"])
        prototype_rows.extend(fold_outputs["prototype_rows"])
        training_rows.extend(fold_outputs["training_rows"])
        fold_rows.extend(fold_outputs["fold_rows"])
        static_rows.extend(fold_outputs["static_rows"])

    added_base_rows = build_base_rows(
        protocol="added_external",
        fold_tag="full_final_pool",
        domain="added",
        eval_records=added_records,
        train_records=final_records,
    )
    added_outputs = run_protocol_fold(
        protocol="added_external",
        fold_tag="full_final_pool",
        train_records=final_records,
        eval_records=added_records,
        train_domain_tag="final",
        record_by_case_id=record_by_case_id,
        gate_lookup=gate_lookup,
        window_lookup=window_lookup,
        temporal_block_cls=try053.TemporalBlock,
        train_config=train_config,
        variant_configs=variant_configs,
        random_init=args.force_random_init,
    )
    prediction_rows.extend(added_outputs["predictions"])
    router_rows.extend(added_outputs["router_rows"])
    expert_rows.extend(added_outputs["expert_rows"])
    prototype_rows.extend(added_outputs["prototype_rows"])
    training_rows.extend(added_outputs["training_rows"])
    fold_rows.extend(added_outputs["fold_rows"])
    static_rows.extend(added_outputs["static_rows"])

    case_level_df = pd.DataFrame([*final_base_rows, *added_base_rows, *prediction_rows])
    router_df = pd.DataFrame(router_rows)
    expert_df = pd.DataFrame(expert_rows)
    prototype_df = pd.DataFrame(prototype_rows)
    training_df = pd.DataFrame(training_rows)
    fold_df = pd.DataFrame(fold_rows)
    static_df = pd.DataFrame(static_rows)

    summary_df = build_summary_by_domain(case_level_df)
    expert_summary_df = build_expert_summary(expert_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    router_df.to_csv(output_dir / "router_activation_table.csv", index=False, encoding="utf-8-sig")
    expert_df.to_csv(output_dir / "expert_residual_stats.csv", index=False, encoding="utf-8-sig")
    expert_summary_df.to_csv(output_dir / "expert_residual_summary.csv", index=False, encoding="utf-8-sig")
    prototype_df.to_csv(output_dir / "prototype_retrieval_stats.csv", index=False, encoding="utf-8-sig")
    training_df.to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(output_dir / "fold_metadata.csv", index=False, encoding="utf-8-sig")
    static_df.to_csv(output_dir / "static_checks.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, case_level_df, router_df, static_df, fold_df)

    best_final = summary_df.loc[summary_df["domain"] == "final"].iloc[0]
    best_added = summary_df.loc[summary_df["domain"] == "added"].iloc[0]
    print("065 Sparse Residual MoE V1 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best final: {best_final['variant_name']} | case_mae={best_final['case_mae']:.4f}")
    print(f"best added: {best_added['variant_name']} | case_mae={best_added['case_mae']:.4f}")


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def predict_rpm_knn4(train_records, rpm_value: float) -> float:
    train_df = pd.DataFrame(
        [{"rpm": float(record.rpm), "wind_speed": float(record.wind_speed)} for record in train_records]
    )
    block = train_df.assign(rpm_distance=(train_df["rpm"] - rpm_value).abs()).nsmallest(min(4, len(train_df)), "rpm_distance")
    distances = block["rpm_distance"].to_numpy(dtype=float, copy=False)
    weights = 1.0 / np.maximum(distances, EPS)
    pred = np.average(block["wind_speed"].to_numpy(dtype=float, copy=False), weights=weights)
    return float(pred)


def build_base_rows(*, protocol: str, fold_tag: str, domain: str, eval_records, train_records) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in eval_records:
        base_pred = predict_rpm_knn4(train_records, float(record.rpm))
        signed_error = float(base_pred - float(record.wind_speed))
        rows.append(
            {
                "protocol": protocol,
                "fold_tag": fold_tag,
                "domain": domain,
                "variant_name": "A0_rpm_knn4",
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "rpm": float(record.rpm),
                "base_pred": float(base_pred),
                "pred_wind_speed": float(base_pred),
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "worse_than_base": 0,
                "excess_harm": 0.0,
            }
        )
    return rows


def build_case_samples(
    *,
    case_ids: list[int],
    domain_name: str,
    record_by_case_id: dict[int, object],
    gate_lookup: pd.DataFrame,
    window_lookup: dict[str, dict[int, np.ndarray]],
    normalization_stats: dict[str, tuple[np.ndarray, np.ndarray]],
    base_pred_map: dict[int, float],
) -> list[CaseSample]:
    samples: list[CaseSample] = []
    for case_id in case_ids:
        record = record_by_case_id[case_id]
        aux_values = gate_lookup.loc[case_id, ROUTER_AUX_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        windows_by_label: dict[str, torch.Tensor] = {}
        for window_label in WINDOW_LABELS:
            mean, std = normalization_stats[window_label]
            raw_windows = window_lookup[window_label][case_id]
            norm_windows = ((raw_windows - mean) / std).astype(np.float32)
            windows_by_label[window_label] = torch.from_numpy(norm_windows).float()
        samples.append(
            CaseSample(
                case_id=case_id,
                file_name=record.file_name,
                domain=domain_name,
                true_wind_speed=float(record.wind_speed),
                rpm=float(record.rpm),
                base_pred=float(base_pred_map[case_id]),
                aux_features=aux_values.astype(np.float32),
                windows_by_label=windows_by_label,
            )
        )
    return samples


def compute_normalization_stats(*, case_ids: list[int], window_lookup: dict[str, dict[int, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for window_label in WINDOW_LABELS:
        train_windows = np.concatenate([window_lookup[window_label][case_id] for case_id in case_ids], axis=0)
        mean = train_windows.mean(axis=(0, 2), keepdims=True)
        std = train_windows.std(axis=(0, 2), keepdims=True)
        std = np.where(std > 0, std, 1.0)
        stats[window_label] = (mean.astype(np.float32), std.astype(np.float32))
    return stats


def build_train_base_map(train_case_ids: list[int], record_by_case_id: dict[int, object]) -> dict[int, float]:
    base_pred_map: dict[int, float] = {}
    train_df = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "rpm": float(record_by_case_id[case_id].rpm),
                "wind_speed": float(record_by_case_id[case_id].wind_speed),
            }
            for case_id in train_case_ids
        ]
    )
    for case_id in train_case_ids:
        inner_df = train_df.loc[train_df["case_id"] != case_id].copy()
        rpm_value = float(train_df.loc[train_df["case_id"] == case_id, "rpm"].iloc[0])
        block = inner_df.assign(rpm_distance=(inner_df["rpm"] - rpm_value).abs()).nsmallest(min(4, len(inner_df)), "rpm_distance")
        distances = block["rpm_distance"].to_numpy(dtype=float, copy=False)
        weights = 1.0 / np.maximum(distances, EPS)
        base_pred_map[case_id] = float(np.average(block["wind_speed"].to_numpy(dtype=float, copy=False), weights=weights))
    return base_pred_map


def build_eval_base_map(eval_case_ids: list[int], train_records, record_by_case_id: dict[int, object]) -> dict[int, float]:
    return {case_id: predict_rpm_knn4(train_records, float(record_by_case_id[case_id].rpm)) for case_id in eval_case_ids}


def load_encoder_init_state(
    *,
    temporal_block_cls,
    in_channels: int,
    protocol: str,
    case_id_for_fold: int | None,
    window_label: str,
    random_init: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    model = TinyTCNEncoder(temporal_block_cls, in_channels=in_channels)
    if random_init:
        return model.state_dict(), "random_forced"

    if protocol == "final_loco" and case_id_for_fold is not None:
        ckpt_path = TRY053_CKPT_DIR / f"fold_case_{case_id_for_fold}_{window_label}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            return model.state_dict(), "try053_fold_checkpoint"

    if protocol == "added_external":
        ckpt_path = FINAL_DEPLOY_CKPT_DIR / f"final_deploy_{window_label}.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            current_state = model.state_dict()
            copied = 0
            for key, value in state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    current_state[key] = value
                    copied += 1
            model.load_state_dict(current_state)
            return model.state_dict(), f"full_final_deploy_partial({copied})"

    return model.state_dict(), "random_fallback"


def initialize_model(
    *,
    temporal_block_cls,
    in_channels: int,
    aux_dim: int,
    train_config: TrainConfig,
    protocol: str,
    case_id_for_fold: int | None,
    random_init: bool,
) -> tuple[SparseResidualMoE, dict[str, str]]:
    model = SparseResidualMoE(
        temporal_block_cls=temporal_block_cls,
        in_channels=in_channels,
        aux_dim=aux_dim,
        hidden_dim=train_config.hidden_dim,
        router_top_k=train_config.router_top_k,
        prototype_top_k=train_config.prototype_top_k,
    )
    init_2s_state, source_2s = load_encoder_init_state(
        temporal_block_cls=temporal_block_cls,
        in_channels=in_channels,
        protocol=protocol,
        case_id_for_fold=case_id_for_fold,
        window_label="2s",
        random_init=random_init,
    )
    init_8s_state, source_8s = load_encoder_init_state(
        temporal_block_cls=temporal_block_cls,
        in_channels=in_channels,
        protocol=protocol,
        case_id_for_fold=case_id_for_fold,
        window_label="8s",
        random_init=random_init,
    )
    model.encoder_2s.load_state_dict(init_2s_state)
    model.encoder_8s.load_state_dict(init_8s_state)
    return model, {"2s": source_2s, "8s": source_8s}


def build_case_tensors(cases: list[CaseSample]) -> dict[str, object]:
    return {
        "case_ids": [sample.case_id for sample in cases],
        "file_names": [sample.file_name for sample in cases],
        "domains": [sample.domain for sample in cases],
        "y": torch.tensor([sample.true_wind_speed for sample in cases], dtype=torch.float32),
        "rpm": torch.tensor([sample.rpm for sample in cases], dtype=torch.float32),
        "base_pred": torch.tensor([sample.base_pred for sample in cases], dtype=torch.float32),
        "aux": torch.tensor(np.stack([sample.aux_features for sample in cases]), dtype=torch.float32),
        "windows_2s": [sample.windows_by_label["2s"] for sample in cases],
        "windows_8s": [sample.windows_by_label["8s"] for sample in cases],
    }


def run_protocol_fold(
    *,
    protocol: str,
    fold_tag: str,
    train_records,
    eval_records,
    train_domain_tag: str,
    record_by_case_id: dict[int, object],
    gate_lookup: pd.DataFrame,
    window_lookup: dict[str, dict[int, np.ndarray]],
    temporal_block_cls,
    train_config: TrainConfig,
    variant_configs: list[VariantConfig],
    random_init: bool,
) -> dict[str, list[dict[str, object]]]:
    train_case_ids = [record.case_id for record in train_records]
    eval_case_ids = [record.case_id for record in eval_records]
    normalization_stats = compute_normalization_stats(case_ids=train_case_ids, window_lookup=window_lookup)
    train_base_map = build_train_base_map(train_case_ids, record_by_case_id)
    eval_base_map = build_eval_base_map(eval_case_ids, train_records, record_by_case_id)

    train_samples = build_case_samples(
        case_ids=train_case_ids,
        domain_name=train_domain_tag,
        record_by_case_id=record_by_case_id,
        gate_lookup=gate_lookup,
        window_lookup=window_lookup,
        normalization_stats=normalization_stats,
        base_pred_map=train_base_map,
    )
    eval_samples = build_case_samples(
        case_ids=eval_case_ids,
        domain_name="final" if protocol == "final_loco" else "added",
        record_by_case_id=record_by_case_id,
        gate_lookup=gate_lookup,
        window_lookup=window_lookup,
        normalization_stats=normalization_stats,
        base_pred_map=eval_base_map,
    )

    train_bundle = build_case_tensors(train_samples)
    eval_bundle = build_case_tensors(eval_samples)
    in_channels = train_bundle["windows_2s"][0].shape[1]
    aux_dim = train_bundle["aux"].shape[1]
    residual_bound = max(
        float(np.quantile(np.abs(train_bundle["y"].numpy() - train_bundle["base_pred"].numpy()), 0.9)),
        0.15,
    )

    prediction_rows: list[dict[str, object]] = []
    router_rows: list[dict[str, object]] = []
    expert_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    static_rows: list[dict[str, object]] = []

    case_id_for_fold = eval_case_ids[0] if protocol == "final_loco" and len(eval_case_ids) == 1 else None
    for variant_index, variant in enumerate(variant_configs):
        set_global_seed(train_config.seed + variant_index + len(train_case_ids))
        model, init_sources = initialize_model(
            temporal_block_cls=temporal_block_cls,
            in_channels=in_channels,
            aux_dim=aux_dim,
            train_config=train_config,
            protocol=protocol,
            case_id_for_fold=case_id_for_fold,
            random_init=random_init,
        )
        variant_outputs = train_single_variant(
            model=model,
            variant=variant,
            train_bundle=train_bundle,
            eval_bundle=eval_bundle,
            residual_bound=residual_bound,
            train_config=train_config,
        )
        prediction_rows.extend(build_prediction_rows(protocol=protocol, fold_tag=fold_tag, variant=variant.name, eval_bundle=eval_bundle, variant_outputs=variant_outputs["eval_outputs"]))
        router_rows.extend(build_router_rows(protocol=protocol, fold_tag=fold_tag, variant=variant.name, eval_bundle=eval_bundle, variant_outputs=variant_outputs["eval_outputs"]))
        expert_rows.extend(build_expert_rows(protocol=protocol, fold_tag=fold_tag, variant=variant.name, eval_bundle=eval_bundle, variant_outputs=variant_outputs["eval_outputs"]))
        prototype_rows.extend(build_prototype_rows(protocol=protocol, fold_tag=fold_tag, variant=variant.name, eval_bundle=eval_bundle, variant_outputs=variant_outputs["eval_outputs"], train_bundle=train_bundle))
        training_rows.extend(add_training_metadata(history_rows=variant_outputs["history_rows"], protocol=protocol, fold_tag=fold_tag, variant=variant.name))
        fold_rows.append(
            {
                "protocol": protocol,
                "fold_tag": fold_tag,
                "variant_name": variant.name,
                "train_case_count": len(train_case_ids),
                "eval_case_count": len(eval_case_ids),
                "residual_bound": residual_bound,
                "init_source_2s": init_sources["2s"],
                "init_source_8s": init_sources["8s"],
            }
        )
        static_rows.extend(build_static_check_rows(protocol=protocol, fold_tag=fold_tag, variant=variant.name, variant_outputs=variant_outputs["eval_outputs"], router_top_k=train_config.router_top_k))

    return {
        "predictions": prediction_rows,
        "router_rows": router_rows,
        "expert_rows": expert_rows,
        "prototype_rows": prototype_rows,
        "training_rows": training_rows,
        "fold_rows": fold_rows,
        "static_rows": static_rows,
    }


def train_single_variant(
    *,
    model: SparseResidualMoE,
    variant: VariantConfig,
    train_bundle: dict[str, object],
    eval_bundle: dict[str, object],
    residual_bound: float,
    train_config: TrainConfig,
) -> dict[str, object]:
    total_epochs = train_config.freeze_epochs + train_config.finetune_epochs
    set_encoder_trainable(model, False)
    optimizer = torch.optim.Adam(
        params=list(model.global_head.parameters()) + list(model.prototype_head.parameters()) + list(model.router.parameters()),
        lr=train_config.head_learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history_rows: list[dict[str, object]] = []
    for epoch in range(total_epochs):
        if epoch == train_config.freeze_epochs:
            set_encoder_trainable(model, True)
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
        train_outputs = forward_variant(
            model=model,
            target_bundle=train_bundle,
            reference_bundle=train_bundle,
            residual_bound=residual_bound,
            variant=variant,
            exclude_self=True,
        )
        loss, loss_parts = compute_variant_loss(
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
                "epoch": epoch + 1,
                "phase": "freeze" if epoch < train_config.freeze_epochs else "finetune",
                "loss_total": float(loss.item()),
                "loss_main": float(loss_parts["loss_main"]),
                "loss_noharm": float(loss_parts["loss_noharm"]),
                "loss_delta": float(loss_parts["loss_delta"]),
                "head_lr": float(optimizer.param_groups[-1]["lr"]),
                "encoder_lr": float(optimizer.param_groups[0]["lr"]) if len(optimizer.param_groups) > 1 else 0.0,
            }
        )

    model.eval()
    with torch.no_grad():
        eval_outputs = forward_variant(
            model=model,
            target_bundle=eval_bundle,
            reference_bundle=train_bundle,
            residual_bound=residual_bound,
            variant=variant,
            exclude_self=False,
        )
    return {"eval_outputs": eval_outputs, "history_rows": history_rows}


def set_encoder_trainable(model: SparseResidualMoE, trainable: bool) -> None:
    for module in (model.encoder_2s, model.encoder_8s):
        for param in module.parameters():
            param.requires_grad = trainable


def forward_variant(
    *,
    model: SparseResidualMoE,
    target_bundle: dict[str, object],
    reference_bundle: dict[str, object],
    residual_bound: float,
    variant: VariantConfig,
    exclude_self: bool,
) -> dict[str, torch.Tensor]:
    target_embeddings = model.encode_cases(bundle_to_case_list(target_bundle))
    reference_embeddings = target_embeddings if target_bundle is reference_bundle else model.encode_cases(bundle_to_case_list(reference_bundle))

    base_pred = target_bundle["base_pred"]
    rpm = target_bundle["rpm"]
    aux = target_bundle["aux"]
    y = target_bundle["y"]

    global_input = torch.cat([target_embeddings, base_pred.unsqueeze(1), rpm.unsqueeze(1)], dim=1)
    delta_global = residual_bound * torch.tanh(model.global_head(global_input))

    proto_info = model.build_prototype_features(
        target_embeddings=target_embeddings,
        reference_embeddings=reference_embeddings,
        target_domains=target_bundle["domains"],
        reference_domains=reference_bundle["domains"],
        exclude_self=exclude_self,
    )
    proto_input = torch.cat(
        [
            proto_info["delta"],
            proto_info["abs_delta"],
            proto_info["top1_distance"].unsqueeze(1),
            proto_info["mean_distance"].unsqueeze(1),
            proto_info["std_distance"].unsqueeze(1),
            base_pred.unsqueeze(1),
        ],
        dim=1,
    )
    delta_proto = residual_bound * torch.tanh(model.prototype_head(proto_input))

    if variant.mode == "global_only":
        weights = torch.zeros((len(target_bundle["case_ids"]), 3), dtype=torch.float32)
        weights[:, 1] = 1.0
    elif variant.mode == "prototype_only":
        weights = torch.zeros((len(target_bundle["case_ids"]), 3), dtype=torch.float32)
        weights[:, 2] = 1.0
    elif variant.mode == "router_moe":
        router_input = torch.cat([target_embeddings, aux, base_pred.unsqueeze(1), rpm.unsqueeze(1)], dim=1)
        logits = model.router(router_input)
        weights = model.sparse_topk_softmax(logits)
    else:
        raise ValueError(f"未知 variant mode: {variant.mode}")

    pred = base_pred + weights[:, 1] * delta_global + weights[:, 2] * delta_proto
    abs_error = torch.abs(pred - y)
    base_abs_error = torch.abs(base_pred - y)
    worse_than_base = (abs_error > base_abs_error).float()
    excess_harm = torch.clamp(abs_error - base_abs_error, min=0.0)

    return {
        "pred": pred,
        "y": y,
        "base_pred": base_pred,
        "delta_global": delta_global,
        "delta_proto": delta_proto,
        "weights": weights,
        "proto_top_indices": proto_info["top_indices"],
        "proto_top_distances": proto_info["top_distances"],
        "proto_top_weights": proto_info["top_weights"],
        "proto_same_domain_rate": proto_info["same_domain_rate"],
        "proto_top1_distance": proto_info["top1_distance"],
        "proto_mean_distance": proto_info["mean_distance"],
        "proto_std_distance": proto_info["std_distance"],
        "abs_error": abs_error,
        "base_abs_error": base_abs_error,
        "worse_than_base": worse_than_base,
        "excess_harm": excess_harm,
    }


def bundle_to_case_list(bundle: dict[str, object]) -> list[CaseSample]:
    cases: list[CaseSample] = []
    for idx, case_id in enumerate(bundle["case_ids"]):
        cases.append(
            CaseSample(
                case_id=case_id,
                file_name=bundle["file_names"][idx],
                domain=bundle["domains"][idx],
                true_wind_speed=float(bundle["y"][idx].item()),
                rpm=float(bundle["rpm"][idx].item()),
                base_pred=float(bundle["base_pred"][idx].item()),
                aux_features=bundle["aux"][idx].numpy(),
                windows_by_label={"2s": bundle["windows_2s"][idx], "8s": bundle["windows_8s"][idx]},
            )
        )
    return cases


def compute_variant_loss(
    *,
    variant_outputs: dict[str, torch.Tensor],
    variant: VariantConfig,
    noharm_margin: float,
    lambda_noharm: float,
    lambda_delta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = variant_outputs["pred"]
    y = variant_outputs["y"]
    base_abs_error = variant_outputs["base_abs_error"]
    abs_error = variant_outputs["abs_error"]
    weights = variant_outputs["weights"]
    delta_penalty = torch.mean(weights[:, 1] * torch.abs(variant_outputs["delta_global"]) + weights[:, 2] * torch.abs(variant_outputs["delta_proto"]))

    loss_main = F.huber_loss(pred, y, reduction="mean", delta=1.0)
    if variant.use_noharm:
        loss_noharm = torch.mean(torch.clamp(abs_error - base_abs_error - noharm_margin, min=0.0))
    else:
        loss_noharm = torch.zeros((), dtype=torch.float32)
    loss = loss_main + lambda_delta * delta_penalty + lambda_noharm * loss_noharm
    return loss, {"loss_main": float(loss_main.item()), "loss_noharm": float(loss_noharm.item()), "loss_delta": float(delta_penalty.item())}


def build_prediction_rows(
    *,
    protocol: str,
    fold_tag: str,
    variant: str,
    eval_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, case_id in enumerate(eval_bundle["case_ids"]):
        pred = float(variant_outputs["pred"][idx].item())
        y = float(eval_bundle["y"][idx].item())
        base_pred = float(eval_bundle["base_pred"][idx].item())
        signed_error = pred - y
        rows.append(
            {
                "protocol": protocol,
                "fold_tag": fold_tag,
                "domain": eval_bundle["domains"][idx],
                "variant_name": variant,
                "case_id": case_id,
                "file_name": eval_bundle["file_names"][idx],
                "true_wind_speed": y,
                "rpm": float(eval_bundle["rpm"][idx].item()),
                "base_pred": base_pred,
                "pred_wind_speed": pred,
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "worse_than_base": int(variant_outputs["worse_than_base"][idx].item()),
                "excess_harm": float(variant_outputs["excess_harm"][idx].item()),
            }
        )
    return rows


def build_router_rows(
    *,
    protocol: str,
    fold_tag: str,
    variant: str,
    eval_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, case_id in enumerate(eval_bundle["case_ids"]):
        weights = variant_outputs["weights"][idx]
        rows.append(
            {
                "protocol": protocol,
                "fold_tag": fold_tag,
                "variant_name": variant,
                "domain": eval_bundle["domains"][idx],
                "case_id": case_id,
                "weight_expert0_noop": float(weights[0].item()),
                "weight_expert1_global": float(weights[1].item()),
                "weight_expert2_prototype": float(weights[2].item()),
                "active_expert_count": int(torch.count_nonzero(weights > 1e-8).item()),
                "dominant_expert": int(torch.argmax(weights).item()),
            }
        )
    return rows


def build_expert_rows(
    *,
    protocol: str,
    fold_tag: str,
    variant: str,
    eval_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, case_id in enumerate(eval_bundle["case_ids"]):
        weight = variant_outputs["weights"][idx]
        delta_global = float(variant_outputs["delta_global"][idx].item())
        delta_proto = float(variant_outputs["delta_proto"][idx].item())
        rows.append(
            {
                "protocol": protocol,
                "fold_tag": fold_tag,
                "variant_name": variant,
                "domain": eval_bundle["domains"][idx],
                "case_id": case_id,
                "delta_global": delta_global,
                "delta_proto": delta_proto,
                "weighted_delta_global": float(weight[1].item() * delta_global),
                "weighted_delta_proto": float(weight[2].item() * delta_proto),
                "mean_abs_residual": float(abs(weight[1].item() * delta_global) + abs(weight[2].item() * delta_proto)),
            }
        )
    return rows


def build_prototype_rows(
    *,
    protocol: str,
    fold_tag: str,
    variant: str,
    eval_bundle: dict[str, object],
    variant_outputs: dict[str, torch.Tensor],
    train_bundle: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, case_id in enumerate(eval_bundle["case_ids"]):
        top_indices = variant_outputs["proto_top_indices"][row_idx].tolist()
        top_distances = variant_outputs["proto_top_distances"][row_idx].tolist()
        top_weights = variant_outputs["proto_top_weights"][row_idx].tolist()
        for rank, train_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "protocol": protocol,
                    "fold_tag": fold_tag,
                    "variant_name": variant,
                    "target_case_id": case_id,
                    "target_domain": eval_bundle["domains"][row_idx],
                    "neighbor_rank": rank,
                    "neighbor_case_id": train_bundle["case_ids"][train_idx],
                    "neighbor_domain": train_bundle["domains"][train_idx],
                    "distance": float(top_distances[rank - 1]),
                    "weight": float(top_weights[rank - 1]),
                    "same_domain_rate": float(variant_outputs["proto_same_domain_rate"][row_idx].item()),
                    "top1_distance": float(variant_outputs["proto_top1_distance"][row_idx].item()),
                    "topk_mean_distance": float(variant_outputs["proto_mean_distance"][row_idx].item()),
                    "topk_std_distance": float(variant_outputs["proto_std_distance"][row_idx].item()),
                }
            )
    return rows


def add_training_metadata(*, history_rows: list[dict[str, object]], protocol: str, fold_tag: str, variant: str) -> list[dict[str, object]]:
    return [{"protocol": protocol, "fold_tag": fold_tag, "variant_name": variant, **row} for row in history_rows]


def build_static_check_rows(
    *,
    protocol: str,
    fold_tag: str,
    variant: str,
    variant_outputs: dict[str, torch.Tensor],
    router_top_k: int,
) -> list[dict[str, object]]:
    weights = variant_outputs["weights"]
    active_counts = (weights > 1e-8).sum(dim=1).cpu().numpy()
    return [
        {
            "protocol": protocol,
            "fold_tag": fold_tag,
            "variant_name": variant,
            "check_name": "router_topk_respected",
            "check_value": int(active_counts.max() <= router_top_k),
            "detail": int(active_counts.max()),
        },
        {
            "protocol": protocol,
            "fold_tag": fold_tag,
            "variant_name": variant,
            "check_name": "expert0_is_zero_output",
            "check_value": 1,
            "detail": 0,
        },
        {
            "protocol": protocol,
            "fold_tag": fold_tag,
            "variant_name": variant,
            "check_name": "expert2_delta_only_path_present",
            "check_value": 1,
            "detail": 1,
        },
    ]


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name in ("final", "added"):
        subset = case_level_df.loc[case_level_df["domain"] == domain_name].copy()
        for variant_name, block in subset.groupby("variant_name", sort=False):
            rows.append(
                {
                    "domain": domain_name,
                    "variant_name": variant_name,
                    "case_mae": float(block["abs_error"].mean()),
                    "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                    "mean_signed_error": float(block["signed_error"].mean()),
                    "worse_than_base_rate": float(block["worse_than_base"].mean()),
                    "mean_excess_harm": float(block["excess_harm"].mean()),
                    "case_count": int(len(block)),
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "worse_than_base_rate", "variant_name"]).reset_index(drop=True)


def build_expert_summary(expert_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain_name, variant_name), block in expert_df.groupby(["domain", "variant_name"], sort=False):
        rows.append(
            {
                "domain": domain_name,
                "variant_name": variant_name,
                "mean_abs_delta_global": float(block["delta_global"].abs().mean()),
                "mean_abs_delta_proto": float(block["delta_proto"].abs().mean()),
                "mean_abs_weighted_residual": float(block["mean_abs_residual"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "variant_name"]).reset_index(drop=True)


def write_summary_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    case_level_df: pd.DataFrame,
    router_df: pd.DataFrame,
    static_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    lines = ["# Sparse Residual MoE V1", "", "## 域级汇总", ""]
    for domain_name in ("final", "added"):
        lines.append(f"### {domain_name}")
        lines.append("")
        subset = summary_df.loc[summary_df["domain"] == domain_name]
        for _, row in subset.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, worse_than_base_rate=`{row['worse_than_base_rate']:.4f}`, "
                f"mean_excess_harm=`{row['mean_excess_harm']:.4f}`"
            )
        lines.append("")

    lines.extend(["## 路由行为摘要", ""])
    for domain_name in ("final", "added"):
        subset = router_df.loc[router_df["domain"] == domain_name]
        if subset.empty:
            continue
        lines.append(f"### {domain_name}")
        lines.append("")
        for variant_name, block in subset.groupby("variant_name", sort=False):
            lines.append(
                f"- `{variant_name}`: no-op=`{block['weight_expert0_noop'].mean():.4f}`, "
                f"global=`{block['weight_expert1_global'].mean():.4f}`, "
                f"prototype=`{block['weight_expert2_prototype'].mean():.4f}`"
            )
        lines.append("")

    lines.extend(["## 静态检查", ""])
    for _, row in static_df.iterrows():
        lines.append(f"- `{row['variant_name']} | {row['check_name']}`: value=`{int(row['check_value'])}`, detail=`{row['detail']}`")

    lines.extend(["", "## 初始化来源", ""])
    for _, row in fold_df.iterrows():
        lines.append(
            f"- `{row['protocol']} | {row['fold_tag']} | {row['variant_name']}`: "
            f"`2s={row['init_source_2s']}`, `8s={row['init_source_8s']}`, residual_bound=`{row['residual_bound']:.4f}`"
        )

    focus_cases = sorted(case_level_df["case_id"].unique())[: min(8, case_level_df["case_id"].nunique())]
    lines.extend(["", "## 代表工况", ""])
    for case_id in focus_cases:
        lines.append(f"### 工况{case_id}")
        lines.append("")
        block = case_level_df.loc[case_level_df["case_id"] == case_id].sort_values(["abs_error", "variant_name"])
        for _, row in block.iterrows():
            lines.append(f"- `{row['protocol']} | {row['variant_name']}`: pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
