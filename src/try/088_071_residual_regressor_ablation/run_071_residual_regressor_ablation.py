from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "088_071_residual_regressor_ablation"
OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
EMBEDDING_TABLE_PATH = REPO_ROOT / "outputs" / "try" / "069_added2_embedding_pca_projection" / "embedding_case_table.csv"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
ENET_ALPHAS = np.logspace(-4, 1, 12)
K_NEIGHBORS = 4
EPS = 1e-6
BASELINE_071_MAE = 0.6161097459937418


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="071 residual regressor ablation.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_df = pd.read_csv(EMBEDDING_TABLE_PATH, encoding="utf-8-sig")
    external_df = (
        embedding_df.loc[embedding_df["raw_source_domain"].isin(["added", "added2"]) & embedding_df["is_labeled"]]
        .copy()
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    embedding_columns = [column for column in external_df.columns if column.startswith("embedding_")]

    all_pred_df = pd.concat(
        [
            run_external_loocv(external_df, embedding_columns, random_seed=args.random_seed),
            run_added_to_added2_transfer(external_df, embedding_columns, random_seed=args.random_seed),
        ],
        ignore_index=True,
    )
    summary_df = build_summary(all_pred_df)
    compare_df = build_added_to_added2_compare(summary_df)

    all_pred_df.to_csv(output_dir / "all_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_protocol.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(output_dir / "added_to_added2_compare_vs_071.csv", index=False, encoding="utf-8-sig")
    write_summary(output_dir / "summary.md", summary_df, compare_df, all_pred_df)

    best = summary_df.loc[summary_df["protocol"] == "added_to_added2"].iloc[0]
    print("088 071 residual regressor ablation finished.")
    print(f"output_dir={output_dir}")
    print(f"best={best['variant_name']} | added_to_added2 case_mae={best['case_mae']:.4f}")


def run_external_loocv(external_df: pd.DataFrame, embedding_columns: list[str], *, random_seed: int) -> pd.DataFrame:
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
                random_seed=random_seed,
            )
        )
    return pd.DataFrame(rows)


def run_added_to_added2_transfer(external_df: pd.DataFrame, embedding_columns: list[str], *, random_seed: int) -> pd.DataFrame:
    train_df = external_df.loc[external_df["raw_source_domain"] == "added"].copy().reset_index(drop=True)
    test_df = external_df.loc[external_df["raw_source_domain"] == "added2"].copy().reset_index(drop=True)
    return pd.DataFrame(
        build_prediction_rows(
            protocol="added_to_added2",
            train_df=train_df,
            test_df=test_df,
            embedding_columns=embedding_columns,
            random_seed=random_seed,
        )
    )


def build_prediction_rows(
    *,
    protocol: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
    random_seed: int,
) -> list[dict[str, object]]:
    residual_targets = compute_internal_oof_residual_targets(train_df)
    base_pred = compute_rpm_knn_predictions(train_df, test_df["rpm"].to_numpy(dtype=float))
    residual_pred_map = fit_predict_residual_variants(
        train_df=train_df,
        test_df=test_df,
        embedding_columns=embedding_columns,
        residual_targets=residual_targets,
        random_seed=random_seed,
    )

    pred_map = {"rpm_knn4": base_pred}
    for variant_name, residual_pred in residual_pred_map.items():
        pred_map[f"rpm_knn4_plus_{variant_name}"] = base_pred + np.asarray(residual_pred, dtype=float)

    rows: list[dict[str, object]] = []
    for variant_name, pred_values in pred_map.items():
        pred_values = np.asarray(pred_values, dtype=float)
        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            pred = float(pred_values[row_idx])
            signed_error = pred - float(row["wind_speed"])
            rows.append(
                {
                    "protocol": protocol,
                    "variant_name": variant_name,
                    "case_id": int(row["case_id"]),
                    "file_name": str(row["file_name"]),
                    "domain": str(row["raw_source_domain"]),
                    "true_wind_speed": float(row["wind_speed"]),
                    "rpm": float(row["rpm"]),
                    "pred_wind_speed": pred,
                    "signed_error": signed_error,
                    "abs_error": abs(signed_error),
                }
            )
    return rows


def fit_predict_residual_variants(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_columns: list[str],
    residual_targets: np.ndarray,
    random_seed: int,
) -> dict[str, np.ndarray]:
    x_train = train_df[embedding_columns].to_numpy(dtype=float)
    x_test = test_df[embedding_columns].to_numpy(dtype=float)
    y = np.asarray(residual_targets, dtype=float)
    n_train = len(train_df)
    cv = max(2, min(3, n_train))
    k_values = sorted(set([1, 2, min(4, n_train)]))
    result: dict[str, np.ndarray] = {
        "residual_zero": np.zeros(len(test_df), dtype=float),
        "residual_mean": np.full(len(test_df), float(np.mean(y)), dtype=float),
    }

    models = {
        "residual_ridge_cv": Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=RIDGE_ALPHAS))]),
        "residual_lasso_cv": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LassoCV(alphas=ENET_ALPHAS, cv=cv, random_state=random_seed, max_iter=20000)),
            ]
        ),
        "residual_elasticnet_cv": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNetCV(alphas=ENET_ALPHAS, l1_ratio=[0.1, 0.5, 0.9], cv=cv, random_state=random_seed, max_iter=20000)),
            ]
        ),
        "residual_kernel_ridge_linear": Pipeline([("scaler", StandardScaler()), ("model", KernelRidge(alpha=1.0, kernel="linear"))]),
        "residual_kernel_ridge_rbf": Pipeline([("scaler", StandardScaler()), ("model", KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01))]),
        "residual_svr_linear": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear", C=1.0, epsilon=0.05))]),
        "residual_svr_rbf": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.05))]),
        "residual_random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_seed,
            min_samples_leaf=1,
            max_features=1.0,
        ),
        "residual_extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_seed,
            min_samples_leaf=1,
            max_features=1.0,
        ),
    }
    for k in k_values:
        models[f"residual_knn{k}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=k, weights="distance")),
            ]
        )
    for n_components in range(1, min(3, n_train - 1, x_train.shape[1]) + 1):
        models[f"residual_pls{n_components}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", PLSRegression(n_components=n_components)),
            ]
        )

    for variant_name, model in models.items():
        try:
            model.fit(x_train, y)
            pred = model.predict(x_test)
            result[variant_name] = np.asarray(pred, dtype=float).reshape(-1)
        except Exception as exc:  # keep the ablation running on tiny folds
            result[f"{variant_name}__failed_{type(exc).__name__}"] = np.full(len(test_df), np.nan, dtype=float)
    return result


def compute_rpm_knn_predictions(train_df: pd.DataFrame, rpm_values: np.ndarray) -> np.ndarray:
    train_rpm = train_df["rpm"].to_numpy(dtype=float, copy=False)
    train_wind = train_df["wind_speed"].to_numpy(dtype=float, copy=False)
    predictions: list[float] = []
    for rpm_value in np.asarray(rpm_values, dtype=float):
        distances = np.abs(train_rpm - rpm_value)
        order = np.argsort(distances)[: min(K_NEIGHBORS, len(train_df))]
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


def build_summary(all_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid_df = all_pred_df.loc[~all_pred_df["pred_wind_speed"].isna()].copy()
    for (protocol, variant_name), block in valid_df.groupby(["protocol", "variant_name"], sort=False):
        true_values = block["true_wind_speed"].to_numpy(dtype=float)
        pred_values = block["pred_wind_speed"].to_numpy(dtype=float)
        signed_error = pred_values - true_values
        rows.append(
            {
                "protocol": protocol,
                "variant_name": variant_name,
                "case_count": int(len(block)),
                "case_mae": float(mean_absolute_error(true_values, pred_values)),
                "case_rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
                "mean_signed_error": float(np.mean(signed_error)),
                "max_abs_error": float(np.max(np.abs(signed_error))),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def build_added_to_added2_compare(summary_df: pd.DataFrame) -> pd.DataFrame:
    result = summary_df.loc[summary_df["protocol"] == "added_to_added2"].copy().reset_index(drop=True)
    result["delta_case_mae_vs_071"] = result["case_mae"] - BASELINE_071_MAE
    result["beats_071"] = result["case_mae"] < BASELINE_071_MAE
    return result.sort_values(["case_mae", "case_rmse", "variant_name"]).reset_index(drop=True)


def write_summary(output_path: Path, summary_df: pd.DataFrame, compare_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    lines = [
        "# 071 residual regressor ablation",
        "",
        "- 状态：`current`",
        "- 首次确认：`2026-04-13`",
        "- 最近复核：`2026-04-13`",
        "- 固定项：`071` embedding、`rpm_knn4` 主干、内部 OOF residual target",
        f"- `071` 对照 case_mae：`{BASELINE_071_MAE:.4f}`",
        "",
        "## added_to_added2 排名",
        "",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
            f"delta_vs_071=`{row['delta_case_mae_vs_071']:.4f}`, beats_071=`{bool(row['beats_071'])}`"
        )
    lines.extend(["", "## external_loocv 排名", ""])
    loocv = summary_df.loc[summary_df["protocol"] == "external_loocv"].copy()
    for _, row in loocv.iterrows():
        lines.append(f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`")
    lines.extend(["", "## added_to_added2 逐工况：Top 5", ""])
    top_variants = compare_df.head(5)["variant_name"].tolist()
    block = pred_df.loc[(pred_df["protocol"] == "added_to_added2") & (pred_df["variant_name"].isin(top_variants))].copy()
    for variant_name in top_variants:
        lines.append(f"### {variant_name}")
        lines.append("")
        for _, row in block.loc[block["variant_name"] == variant_name].sort_values("case_id").iterrows():
            lines.append(
                f"- `工况{int(row['case_id'])}`: pred=`{float(row['pred_wind_speed']):.4f}`, "
                f"true=`{float(row['true_wind_speed']):.4f}`, abs_error=`{float(row['abs_error']):.4f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
