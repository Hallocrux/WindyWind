from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "072_external_embedding_topk_loocv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
EMBEDDING_TABLE_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_case_table.csv"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RPM_K = 4
TOP_K_VALUES = (3, 4)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 added+added2 external LOOCV 下验证 embedding top-k residual。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    embedding_df = pd.read_csv(EMBEDDING_TABLE_PATH, encoding="utf-8-sig")
    external_df = (
        embedding_df.loc[embedding_df["raw_source_domain"].isin(["added", "added2"]) & embedding_df["is_labeled"]]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    embedding_columns = [column for column in external_df.columns if column.startswith("embedding_")]

    case_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for holdout_case_id in external_df["case_id"].tolist():
        train_df = external_df.loc[external_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_df = external_df.loc[external_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        fold_case_rows, fold_neighbor_rows = run_one_fold(train_df, test_df, embedding_columns)
        case_rows.extend(fold_case_rows)
        neighbor_rows.extend(fold_neighbor_rows)

    case_level_df = pd.DataFrame(case_rows).sort_values(["variant_name", "case_id"]).reset_index(drop=True)
    neighbor_df = pd.DataFrame(neighbor_rows).sort_values(["holdout_case_id", "variant_name", "neighbor_rank"]).reset_index(drop=True)
    summary_by_variant_df = build_summary_by_variant(case_level_df)
    summary_by_domain_df = build_summary_by_domain(case_level_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_variant_df.to_csv(output_dir / "summary_by_variant.csv", index=False, encoding="utf-8-sig")
    summary_by_domain_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")
    neighbor_df.to_csv(output_dir / "neighbor_table.csv", index=False, encoding="utf-8-sig")

    create_pred_vs_true_plot(case_level_df, plot_dir / "pred_vs_true.png")
    write_summary_markdown(output_dir / "summary.md", summary_by_variant_df, summary_by_domain_df)

    best_row = summary_by_variant_df.iloc[0]
    print("072 external embedding top-k LOOCV 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"best variant: {best_row['variant_name']} | case_mae={best_row['case_mae']:.4f}")


def run_one_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    residual_targets = compute_internal_oof_residual_targets(train_df)
    base_pred = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))[0]
    embedding_ridge_pred = fit_predict_embedding_ridge(train_df, test_df, embedding_columns)[0]
    embedding_knn_pred = fit_predict_embedding_knn(train_df, test_df, embedding_columns)[0]
    global_residual_pred = fit_predict_embedding_residual_ridge(train_df, test_df, embedding_columns, residual_targets)[0]

    case_rows = [
        build_case_row("rpm_knn4", test_df.iloc[0], base_pred),
        build_case_row("embedding_ridge", test_df.iloc[0], float(embedding_ridge_pred)),
        build_case_row("embedding_knn4", test_df.iloc[0], float(embedding_knn_pred)),
        build_case_row("rpm_knn4_plus_global_embedding_residual_ridge", test_df.iloc[0], float(base_pred + global_residual_pred)),
    ]
    neighbor_rows: list[dict[str, object]] = []

    for top_k in TOP_K_VALUES:
        topk_df, topk_neighbor_rows = select_topk_neighbors(train_df, test_df, embedding_columns, residual_targets, top_k=top_k)
        neighbor_rows.extend(topk_neighbor_rows)

        mean_residual = compute_weighted_mean_residual(topk_df)
        case_rows.append(
            build_case_row(
                f"rpm_knn4_plus_topk{top_k}_residual_mean",
                test_df.iloc[0],
                float(base_pred + mean_residual),
            )
        )

        ridge_residual = fit_predict_local_residual_ridge(topk_df, test_df, embedding_columns)[0]
        case_rows.append(
            build_case_row(
                f"rpm_knn4_plus_topk{top_k}_residual_ridge",
                test_df.iloc[0],
                float(base_pred + ridge_residual),
            )
        )

    return case_rows, neighbor_rows


def build_case_row(variant_name: str, row: pd.Series, pred_value: float) -> dict[str, object]:
    signed_error = float(pred_value - float(row["wind_speed"]))
    return {
        "variant_name": variant_name,
        "case_id": int(row["case_id"]),
        "file_name": str(row["file_name"]),
        "domain": str(row["raw_source_domain"]),
        "true_wind_speed": float(row["wind_speed"]),
        "rpm": float(row["rpm"]),
        "pred_wind_speed": float(pred_value),
        "signed_error": signed_error,
        "abs_error": abs(signed_error),
    }


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[float] = []
    for rpm_value in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm_value)
        order = np.argsort(distances)
        k = min(RPM_K, len(order))
        order = order[:k]
        weights = 1.0 / np.maximum(distances[order], EPS)
        weights = weights / weights.sum()
        predictions.append(float(np.dot(weights, train_wind[order])))
    return np.asarray(predictions, dtype=float)


def compute_internal_oof_residual_targets(train_df: pd.DataFrame) -> np.ndarray:
    if len(train_df) <= 1:
        return np.zeros(len(train_df), dtype=float)
    rpm_values = train_df["rpm"].to_numpy(dtype=float, copy=False)
    true_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    residuals: list[float] = []
    for row_idx in range(len(train_df)):
        inner_train = train_df.drop(index=train_df.index[row_idx]).reset_index(drop=True)
        pred = compute_rpm_knn_predictions(inner_train, np.asarray([rpm_values[row_idx]], dtype=float))[0]
        residuals.append(float(true_wind[row_idx] - pred))
    return np.asarray(residuals, dtype=float)


def fit_predict_embedding_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame, embedding_columns: list[str]) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    model.fit(train_df[embedding_columns].to_numpy(dtype=float), train_df["wind_speed"].to_numpy(dtype=float))
    return model.predict(test_df[embedding_columns].to_numpy(dtype=float))


def fit_predict_embedding_knn(train_df: pd.DataFrame, test_df: pd.DataFrame, embedding_columns: list[str]) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=min(4, len(train_df)), weights="distance")),
        ]
    )
    model.fit(train_df[embedding_columns].to_numpy(dtype=float), train_df["wind_speed"].to_numpy(dtype=float))
    return model.predict(test_df[embedding_columns].to_numpy(dtype=float))


def fit_predict_embedding_residual_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
    residual_targets: np.ndarray,
) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    model.fit(train_df[embedding_columns].to_numpy(dtype=float), np.asarray(residual_targets, dtype=float))
    return model.predict(test_df[embedding_columns].to_numpy(dtype=float))


def select_topk_neighbors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
    residual_targets: np.ndarray,
    *,
    top_k: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train_matrix = train_df[embedding_columns].to_numpy(dtype=float, copy=False)
    test_matrix = test_df[embedding_columns].to_numpy(dtype=float, copy=False)
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    train_scaled = (train_matrix - mean) / std
    test_scaled = (test_matrix - mean) / std

    distances = np.linalg.norm(train_scaled - test_scaled[0], axis=1)
    order = np.argsort(distances)
    top_k = min(top_k, len(order))
    top_indices = order[:top_k]

    topk_df = train_df.iloc[top_indices].copy().reset_index(drop=True)
    topk_df["distance"] = distances[top_indices]
    topk_df["distance_weight"] = 1.0 / np.maximum(topk_df["distance"].to_numpy(dtype=float), EPS)
    topk_df["distance_weight"] = topk_df["distance_weight"] / topk_df["distance_weight"].sum()
    topk_df["rpm_residual_oof"] = residual_targets[top_indices]

    neighbor_rows = []
    for rank, (_, row) in enumerate(topk_df.iterrows(), start=1):
        neighbor_rows.append(
            {
                "holdout_case_id": int(test_df["case_id"].iloc[0]),
                "holdout_domain": str(test_df["raw_source_domain"].iloc[0]),
                "variant_name": f"topk{top_k}",
                "neighbor_rank": rank,
                "neighbor_case_id": int(row["case_id"]),
                "neighbor_domain": str(row["raw_source_domain"]),
                "distance": float(row["distance"]),
                "distance_weight": float(row["distance_weight"]),
                "rpm_residual_oof": float(row["rpm_residual_oof"]),
            }
        )
    return topk_df, neighbor_rows


def compute_weighted_mean_residual(topk_df: pd.DataFrame) -> float:
    weights = topk_df["distance_weight"].to_numpy(dtype=float, copy=False)
    residuals = topk_df["rpm_residual_oof"].to_numpy(dtype=float, copy=False)
    return float(np.dot(weights, residuals))


def fit_predict_local_residual_ridge(topk_df: pd.DataFrame, test_df: pd.DataFrame, embedding_columns: list[str]) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    model.fit(topk_df[embedding_columns].to_numpy(dtype=float), topk_df["rpm_residual_oof"].to_numpy(dtype=float))
    return model.predict(test_df[embedding_columns].to_numpy(dtype=float))


def summarize_block(block: pd.DataFrame) -> dict[str, object]:
    true_values = block["true_wind_speed"].to_numpy(dtype=float)
    pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    return {
        "case_count": int(len(block)),
        "case_mae": float(mean_absolute_error(true_values, pred_values)),
        "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mean_signed_error": float(np.mean(signed_error)),
        "max_abs_error": float(np.max(np.abs(signed_error))),
    }


def build_summary_by_variant(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_level_df.groupby("variant_name", sort=False):
        row = {"variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, variant_name), block in case_level_df.groupby(["domain", "variant_name"], sort=False):
        row = {"domain": domain, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def create_pred_vs_true_plot(case_level_df: pd.DataFrame, output_path: Path) -> None:
    variants = summary_order(case_level_df)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    color_map = {"added": "#f58518", "added2": "#e45756"}
    for ax, variant_name in zip(axes, variants):
        block = case_level_df.loc[case_level_df["variant_name"] == variant_name].copy()
        for domain_name, domain_block in block.groupby("domain", sort=True):
            ax.scatter(
                domain_block["true_wind_speed"],
                domain_block["pred_wind_speed"],
                s=100,
                alpha=0.9,
                color=color_map.get(domain_name, "#4c78a8"),
                label=domain_name,
            )
        lo = min(block["true_wind_speed"].min(), block["pred_wind_speed"].min()) - 0.2
        hi = max(block["true_wind_speed"].max(), block["pred_wind_speed"].max()) + 0.2
        ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1.5)
        ax.set_title(variant_name)
        ax.set_xlabel("true_wind_speed")
        ax.set_ylabel("pred_wind_speed")
        ax.legend()
    for ax in axes[len(variants):]:
        ax.axis("off")
    fig.suptitle("External LOOCV with top-k selection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summary_order(case_level_df: pd.DataFrame) -> list[str]:
    summary_df = build_summary_by_variant(case_level_df)
    return summary_df["variant_name"].tolist()[:6]


def write_summary_markdown(output_path: Path, summary_by_variant_df: pd.DataFrame, summary_by_domain_df: pd.DataFrame) -> None:
    lines = ["# external embedding top-k LOOCV", "", "## Summary By Variant", ""]
    for _, row in summary_by_variant_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
            f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
        )
    lines.extend(["", "## Summary By Domain", ""])
    for domain, block in summary_by_domain_df.groupby("domain", sort=False):
        lines.append(f"### {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
