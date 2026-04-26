from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "070_added_added2_pca_line_regression"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
PCA_COORD_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_pca_coords.csv"

DOMAIN_COLORS = {
    "added": "#f58518",
    "added2": "#e45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="沿 added+added2 的 PCA 主轴做一维回归 quickcheck。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pca_df = pd.read_csv(PCA_COORD_PATH, encoding="utf-8-sig")
    ext_df = (
        pca_df.loc[pca_df["raw_source_domain"].isin(["added", "added2"]) & pca_df["is_labeled"]]
        .copy()
        .reset_index(drop=True)
    )

    projected_df, line_meta = add_line_projection(ext_df)
    loocv_df = build_loocv_summary(projected_df)
    transfer_df = build_added_train_added2_test_summary(projected_df)

    projected_df.to_csv(output_dir / "added_added2_with_line_projection.csv", index=False, encoding="utf-8-sig")
    loocv_df.to_csv(output_dir / "loocv_summary.csv", index=False, encoding="utf-8-sig")
    transfer_df.to_csv(output_dir / "added_train_added2_test_summary.csv", index=False, encoding="utf-8-sig")

    create_pca_line_projection_plot(projected_df, line_meta, plot_dir / "pca_line_projection.png")
    create_projection_vs_wind_plot(projected_df, plot_dir / "projection_vs_wind.png")
    write_summary_markdown(output_dir / "summary.md", projected_df, loocv_df, transfer_df, line_meta)

    print("070 added+added2 PCA line regression 已完成。")
    print(f"输出目录: {output_dir}")


def add_line_projection(ext_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    coords = ext_df[["pca1", "pca2"]].to_numpy(dtype=float, copy=False)
    center = coords.mean(axis=0)
    shifted = coords - center
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    direction = vh[0].astype(float)
    if direction[0] < 0:
        direction = -direction
    projection = shifted @ direction
    orthogonal = shifted @ np.array([-direction[1], direction[0]], dtype=float)

    result = ext_df.copy()
    result["line_coord"] = projection
    result["line_orth_dist_signed"] = orthogonal
    result["line_orth_dist_abs"] = np.abs(orthogonal)
    return result, {
        "center_pca1": float(center[0]),
        "center_pca2": float(center[1]),
        "direction_pca1": float(direction[0]),
        "direction_pca2": float(direction[1]),
    }


def fit_predict_line_linear(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_df[["line_coord"]].to_numpy(dtype=float), train_df["wind_speed"].to_numpy(dtype=float))
    return model.predict(test_df[["line_coord"]].to_numpy(dtype=float))


def fit_predict_line_quadratic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(train_df[["line_coord"]].to_numpy(dtype=float), train_df["wind_speed"].to_numpy(dtype=float))
    return model.predict(test_df[["line_coord"]].to_numpy(dtype=float))


def fit_predict_pca2d_linear(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_df[["pca1", "pca2"]].to_numpy(dtype=float), train_df["wind_speed"].to_numpy(dtype=float))
    return model.predict(test_df[["pca1", "pca2"]].to_numpy(dtype=float))


def summarize_prediction_frame(pred_df: pd.DataFrame, variant_name: str) -> dict[str, object]:
    true_values = pred_df["wind_speed"].to_numpy(dtype=float)
    pred_values = pred_df["pred_wind_speed"].to_numpy(dtype=float)
    signed_error = pred_values - true_values
    return {
        "variant_name": variant_name,
        "case_count": int(len(pred_df)),
        "case_mae": float(mean_absolute_error(true_values, pred_values)),
        "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "mean_signed_error": float(np.mean(signed_error)),
        "max_abs_error": float(np.max(np.abs(signed_error))),
    }


def build_loocv_summary(projected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for variant_name, predictor in (
        ("line_linear_loocv", fit_predict_line_linear),
        ("line_quadratic_loocv", fit_predict_line_quadratic),
        ("pca2d_linear_loocv", fit_predict_pca2d_linear),
    ):
        fold_rows: list[dict[str, object]] = []
        for case_id in projected_df["case_id"].tolist():
            train_df = projected_df.loc[projected_df["case_id"] != case_id].copy()
            test_df = projected_df.loc[projected_df["case_id"] == case_id].copy()
            pred = predictor(train_df, test_df)
            fold_rows.append(
                {
                    "case_id": int(test_df["case_id"].iloc[0]),
                    "raw_source_domain": str(test_df["raw_source_domain"].iloc[0]),
                    "wind_speed": float(test_df["wind_speed"].iloc[0]),
                    "pred_wind_speed": float(pred[0]),
                }
            )
        fold_df = pd.DataFrame(fold_rows)
        summary = summarize_prediction_frame(fold_df, variant_name)
        rows.append(summary)
        fold_df["variant_name"] = variant_name
        prediction_rows.append(fold_df)

    prediction_df = pd.concat(prediction_rows, ignore_index=True)
    prediction_df.to_csv(OUTPUT_DIR / "loocv_case_predictions.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows).sort_values("case_mae").reset_index(drop=True)


def build_added_train_added2_test_summary(projected_df: pd.DataFrame) -> pd.DataFrame:
    train_df = projected_df.loc[projected_df["raw_source_domain"] == "added"].copy()
    test_df = projected_df.loc[projected_df["raw_source_domain"] == "added2"].copy()
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for variant_name, predictor in (
        ("line_linear_added_to_added2", fit_predict_line_linear),
        ("line_quadratic_added_to_added2", fit_predict_line_quadratic),
        ("pca2d_linear_added_to_added2", fit_predict_pca2d_linear),
    ):
        pred = predictor(train_df, test_df)
        pred_df = test_df[["case_id", "raw_source_domain", "wind_speed"]].copy()
        pred_df["pred_wind_speed"] = pred.astype(float)
        rows.append(summarize_prediction_frame(pred_df, variant_name))
        pred_df["variant_name"] = variant_name
        prediction_rows.append(pred_df)

    prediction_df = pd.concat(prediction_rows, ignore_index=True)
    prediction_df.to_csv(OUTPUT_DIR / "added_train_added2_test_case_predictions.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows).sort_values("case_mae").reset_index(drop=True)


def create_pca_line_projection_plot(projected_df: pd.DataFrame, line_meta: dict[str, object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for domain_name in ("added", "added2"):
        block = projected_df.loc[projected_df["raw_source_domain"] == domain_name]
        ax.scatter(
            block["pca1"],
            block["pca2"],
            s=130,
            alpha=0.9,
            color=DOMAIN_COLORS[domain_name],
            label=domain_name,
        )
    for _, row in projected_df.iterrows():
        ax.text(row["pca1"], row["pca2"], str(int(row["case_id"])), fontsize=9, ha="left", va="bottom")

    center = np.array([line_meta["center_pca1"], line_meta["center_pca2"]], dtype=float)
    direction = np.array([line_meta["direction_pca1"], line_meta["direction_pca2"]], dtype=float)
    span = projected_df["line_coord"].abs().max() + 2.0
    line_points = np.vstack([center - span * direction, center + span * direction])
    ax.plot(line_points[:, 0], line_points[:, 1], color="#444444", linewidth=2.0, linestyle="--", label="principal line")

    ax.set_title("Added + Added2 PCA principal line")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_projection_vs_wind_plot(projected_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for domain_name in ("added", "added2"):
        block = projected_df.loc[projected_df["raw_source_domain"] == domain_name].sort_values("line_coord")
        ax.scatter(
            block["line_coord"],
            block["wind_speed"],
            s=130,
            alpha=0.9,
            color=DOMAIN_COLORS[domain_name],
            label=domain_name,
        )
        for _, row in block.iterrows():
            ax.text(row["line_coord"], row["wind_speed"], str(int(row["case_id"])), fontsize=9, ha="left", va="bottom")

    line_x = np.linspace(projected_df["line_coord"].min() - 0.5, projected_df["line_coord"].max() + 0.5, 200).reshape(-1, 1)
    linear_model = LinearRegression().fit(projected_df[["line_coord"]].to_numpy(dtype=float), projected_df["wind_speed"].to_numpy(dtype=float))
    quad_model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("linear", LinearRegression())])
    quad_model.fit(projected_df[["line_coord"]].to_numpy(dtype=float), projected_df["wind_speed"].to_numpy(dtype=float))
    ax.plot(line_x[:, 0], linear_model.predict(line_x), color="#222222", linewidth=2.0, label="linear fit")
    ax.plot(line_x[:, 0], quad_model.predict(line_x), color="#777777", linewidth=2.0, linestyle="--", label="quadratic fit")

    ax.set_title("Wind speed vs principal-line coordinate")
    ax.set_xlabel("principal-line coordinate")
    ax.set_ylabel("wind_speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    projected_df: pd.DataFrame,
    loocv_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    line_meta: dict[str, object],
) -> None:
    lines = [
        "# added added2 PCA line regression",
        "",
        "## 主轴信息",
        "",
        f"- center = (`{line_meta['center_pca1']:.4f}`, `{line_meta['center_pca2']:.4f}`)",
        f"- direction = (`{line_meta['direction_pca1']:.4f}`, `{line_meta['direction_pca2']:.4f}`)",
        f"- 最大垂距 = `{projected_df['line_orth_dist_abs'].max():.4f}`",
        f"- 平均垂距 = `{projected_df['line_orth_dist_abs'].mean():.4f}`",
        "",
        "## LOOCV",
        "",
    ]
    for _, row in loocv_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
            f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
        )
    lines.extend(["", "## added -> added2 外推", ""])
    for _, row in transfer_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
            f"case_rmse=`{row['case_rmse']:.4f}`, mean_signed_error=`{row['mean_signed_error']:.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
