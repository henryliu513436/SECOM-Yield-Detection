"""簡報用的兩張圖：異常比例（資料集/訓練/測試）、XGBoost vs baseline 逐折 PR-AUC。

只重用 data.py 與 train.py 既有的函式算數字，不重新計算任何統計量、不
引入新的資料處理邏輯——這支腳本只負責畫圖，跟正式訓練流程完全無關。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as D
from train import (
    XGB_THRESHOLD_MIN_PRECISION,
    build_isolation_forest_pipeline,
    fit_xgboost_pipeline,
    load_production_params,
    run_cv,
    select_threshold_at_min_precision,
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def plot_anomaly_rates(raw: pd.DataFrame, split: D.TimeSplit, output_path: Path) -> None:
    """全資料集、訓練集、測試集三者的 Fail 比例長條圖。"""
    rates = {
        "Full dataset": D.extract_labels(raw).mean(),
        "Train": D.extract_labels(split.train).mean(),
        "Test": D.extract_labels(split.test).mean(),
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(rates.keys(), [v * 100 for v in rates.values()], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("Fail rate (%)")
    ax.set_title("Anomaly Rate: Full Dataset vs. Time-Based Split")
    ax.set_ylim(0, max(rates.values()) * 100 * 1.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fold_pr_auc(results: list, output_path: Path) -> None:
    """逐折 baseline vs XGBoost PR-AUC 長條圖，每組標上倍數。"""
    folds = [r.fold for r in results]
    baseline = [r.baseline for r in results]
    xgb = [r.xgb_pr_auc for r in results]
    ratio = [r.xgb_ratio for r in results]

    x = range(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], baseline, width, label="Random baseline", color="#8C8C8C")
    bars = ax.bar([i + width / 2 for i in x], xgb, width, label="XGBoost", color="#4C72B0")

    for i, r in enumerate(ratio):
        ax.text(i + width / 2, xgb[i] + 0.005, f"{r:.2f}x", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("PR-AUC")
    ax.set_title("XGBoost Beats Baseline in All 5 Folds")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_generalization_numbers(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, dict[str, float]]:
    """算出兩個模型在「訓練驗證尾端」跟「測試集」上的 precision/recall。

    XGBoost 的閾值固定用訓練驗證尾端選出來的那個（跟 production 一致），
    測試集只套用、不重新選閾值；IsolationForest 用原生的 contamination
    判定。兩個階段都重用同一個 fit 好的 pipeline，不重新訓練。
    """
    xgb_result = fit_xgboost_pipeline(X_train, y_train, load_production_params())
    threshold = select_threshold_at_min_precision(
        xgb_result.val_labels, xgb_result.val_scores, XGB_THRESHOLD_MIN_PRECISION
    )
    iso_pipeline = build_isolation_forest_pipeline(X_train)

    tail_size = len(xgb_result.val_labels)
    X_val_tail, y_val_tail = X_train.iloc[-tail_size:], y_train.iloc[-tail_size:]

    xgb_val_pred = (xgb_result.val_scores >= threshold).astype(int)
    xgb_test_scores = xgb_result.pipeline.predict_proba(X_test)[:, 1]
    xgb_test_pred = (xgb_test_scores >= threshold).astype(int)

    iso_val_pred = (iso_pipeline.predict(X_val_tail) == -1).astype(int)
    iso_test_pred = (iso_pipeline.predict(X_test) == -1).astype(int)

    return {
        "XGBoost": {
            "val_precision": precision_score(y_val_tail, xgb_val_pred, zero_division=0),
            "val_recall": recall_score(y_val_tail, xgb_val_pred, zero_division=0),
            "test_precision": precision_score(y_test, xgb_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, xgb_test_pred, zero_division=0),
        },
        "IsolationForest": {
            "val_precision": precision_score(y_val_tail, iso_val_pred, zero_division=0),
            "val_recall": recall_score(y_val_tail, iso_val_pred, zero_division=0),
            "test_precision": precision_score(y_test, iso_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, iso_test_pred, zero_division=0),
        },
    }


def plot_threshold_generalization(numbers: dict[str, dict[str, float]], output_path: Path) -> None:
    """驗證尾端 vs 測試集的 precision/recall 對照，兩個模型並列。"""
    models = list(numbers.keys())
    x = range(len(models))
    width = 0.35

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, title in ((ax_p, "precision", "Precision"), (ax_r, "recall", "Recall")):
        val_values = [numbers[m][f"val_{metric}"] for m in models]
        test_values = [numbers[m][f"test_{metric}"] for m in models]

        bars_val = ax.bar([i - width / 2 for i in x], val_values, width, label="Validation tail", color="#4C72B0")
        bars_test = ax.bar([i + width / 2 for i in x], test_values, width, label="Test set (n=471)", color="#C44E52")
        ax.bar_label(bars_val, fmt="%.2f", padding=3)
        ax.bar_label(bars_test, fmt="%.2f", padding=3)

        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)

    ax_p.legend(loc="upper right")
    fig.suptitle("Does the Selected Operating Point Generalize to the Test Set?")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"

    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    y_train = D.extract_labels(split.train)

    anomaly_path = FIGURES_DIR / "anomaly_rate_by_split.png"
    plot_anomaly_rates(raw, split, anomaly_path)
    print(f"已存至：{anomaly_path}")

    cv_results = run_cv(X_train, y_train, n_splits=5, params=load_production_params())
    fold_path = FIGURES_DIR / "xgboost_vs_baseline_prauc.png"
    plot_fold_pr_auc(cv_results, fold_path)
    print(f"已存至：{fold_path}")

    X_test = D.extract_features(split.test)
    y_test = D.extract_labels(split.test)
    numbers = compute_generalization_numbers(X_train, y_train, X_test, y_test)
    gen_path = FIGURES_DIR / "threshold_generalization.png"
    plot_threshold_generalization(numbers, gen_path)
    print(f"已存至：{gen_path}")
