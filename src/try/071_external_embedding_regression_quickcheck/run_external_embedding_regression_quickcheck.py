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
TRY_NAME = "071_external_embedding_regression_quickcheck"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
EMBEDDING_TABLE_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_case_table.csv"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
K_NEIGHBORS = 4
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 added+added2 上比较高维 embedding 回归 quickcheck。")
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

    loocv_pred_df = run_external_loocv(external_df, embedding_columns)
    transfer_pred_df = run_added_to_added2_transfer(external_df, embedding_columns)
    all_pred_df = pd.concat([loocv_pred_df, transfer_pred_df], ignore_index=True)

    summary_by_protocol_df = build_summary_by_protocol(all_pred_df)
    summary_by_protocol_and_domain_df = build_summary_by_protocol_and_domain(all_pred_df)

    all_pred_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    summary_by_protocol_and_domain_df.to_csv(output_dir / "summary_by_protocol_and_domain.csv", index=False, encoding="utf-8-sig")

    create_pred_vs_true_plot(
        all_pred_df.loc[all_pred_df["protocol"] == "external_loocv"].copy(),
        plot_dir / "pred_vs_true_external_loocv.png",
        "External LOOCV",
    )
    create_pred_vs_true_plot(
        all_pred_df.loc[all_pred_df["protocol"] == "added_to_added2"].copy(),
        plot_dir / "pred_vs_true_added_to_added2.png",
        "Added to Added2",
    )
    write_summary_markdown(output_dir / "summary.md", summary_by_protocol_df, summary_by_protocol_and_domain_df)

    best_row = summary_by_protocol_df.iloc[0]
    print("071 external embedding regression quickcheck 已完成。")
    print(f"输出目录: {output_dir}")
    print(
        f"best protocol={best_row['protocol']} | "
        f"variant={best_row['variant_name']} | "
        f"case_mae={best_row['case_mae']:.4f}"
    )


def run_external_loocv(external_df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout_case_id in external_df["case_id"].tolist():
        train_df = external_df.loc[external_df["case_id"] != holdout_case_id].copy().reset_index(drop=True)
        test_df = external_df.loc[external_df["case_id"] == holdout_case_id].copy().reset_index(drop=True)
        rows.extend(
            build_prediction_rows(
                protocol="external_loocv",
                train_df=train_df,
                test_df=test_df,
                embedding_columns=embedding_columns,
            )
        )
    return pd.DataFrame(rows)


def run_added_to_added2_transfer(external_df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    train_df = external_df.loc[external_df["raw_source_domain"] == "added"].copy().reset_index(drop=True)
    test_df = external_df.loc[external_df["raw_source_domain"] == "added2"].copy().reset_index(drop=True)
    rows = build_prediction_rows(
        protocol="added_to_added2",
        train_df=train_df,
        test_df=test_df,
        embedding_columns=embedding_columns,
    )
    return pd.DataFrame(rows)


def build_prediction_rows(
    *,
    protocol: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
) -> list[dict[str, object]]:
    residual_targets = compute_internal_oof_residual_targets(train_df)
    base_test_pred = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))

    pred_map = {
        "rpm_knn4": base_test_pred,
        "embedding_ridge": fit_predict_embedding_ridge(train_df, test_df, embedding_columns),
        "embedding_knn4": fit_predict_embedding_knn(train_df, test_df, embedding_columns),
        "rpm_knn4_plus_embedding_residual_ridge": base_test_pred
        + fit_predict_embedding_residual_ridge(train_df, test_df, embedding_columns, residual_targets),
    }

    rows: list[dict[str, object]] = []
    for variant_name, pred_values in pred_map.items():
        pred_values = np.asarray(pred_values, dtype=float)
        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            pred_value = float(pred_values[row_idx])
            signed_error = pred_value - float(row["wind_speed"])
            rows.append(
                {
                    "protocol": protocol,
                    "variant_name": variant_name,
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["raw_source_domain"]),
                    "true_wind_speed": float(row["wind_speed"]),
                    "rpm": float(row["rpm"]),
                    "pred_wind_speed": pred_value,
                    "signed_error": signed_error,
                    "abs_error": abs(signed_error),
                }
            )
    return rows


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[float] = []
    for rpm_value in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm_value)
        order = np.argsort(distances)
        k = min(K_NEIGHBORS, len(order))
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
    model.fit(
        train_df[embedding_columns].to_numpy(dtype=float),
        train_df["wind_speed"].to_numpy(dtype=float),
    )
    return model.predict(test_df[embedding_columns].to_numpy(dtype=float))


def fit_predict_embedding_knn(train_df: pd.DataFrame, test_df: pd.DataFrame, embedding_columns: list[str]) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=min(K_NEIGHBORS, len(train_df)), weights="distance")),
        ]
    )
    model.fit(
        train_df[embedding_columns].to_numpy(dtype=float),
        train_df["wind_speed"].to_numpy(dtype=float),
    )
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
    model.fit(
        train_df[embedding_columns].to_numpy(dtype=float),
        np.asarray(residual_targets, dtype=float),
    )
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


def build_summary_by_protocol(all_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, variant_name), block in all_pred_df.groupby(["protocol", "variant_name"], sort=False):
        row = {"protocol": protocol, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_summary_by_protocol_and_domain(all_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, domain, variant_name), block in all_pred_df.groupby(["protocol", "domain", "variant_name"], sort=False):
        row = {"protocol": protocol, "domain": domain, "variant_name": variant_name}
        row.update(summarize_block(block))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "domain", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def create_pred_vs_true_plot(pred_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if pred_df.empty:
        return
    variants = sorted(pred_df["variant_name"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    domain_color = {"added": "#f58518", "added2": "#e45756"}
    for ax, variant_name in zip(axes, variants):
        block = pred_df.loc[pred_df["variant_name"] == variant_name].copy()
        for domain_name, domain_block in block.groupby("domain", sort=True):
            ax.scatter(
                domain_block["true_wind_speed"],
                domain_block["pred_wind_speed"],
                s=100,
                alpha=0.9,
                color=domain_color.get(domain_name, "#4c78a8"),
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
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    summary_by_protocol_df: pd.DataFrame,
    summary_by_protocol_and_domain_df: pd.DataFrame,
) -> None:
    lines = ["# external embedding regression quickcheck", "", "## Summary By Protocol", ""]
    for protocol, protocol_df in summary_by_protocol_df.groupby("protocol", sort=False):
        lines.append(f"### {protocol}")
        lines.append("")
        for _, row in protocol_df.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
            )
        lines.append("")
    lines.append("## Summary By Protocol And Domain")
    lines.append("")
    for (protocol, domain), block in summary_by_protocol_and_domain_df.groupby(["protocol", "domain"], sort=False):
        lines.append(f"### {protocol} | {domain}")
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
