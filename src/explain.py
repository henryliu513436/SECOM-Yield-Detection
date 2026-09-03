"""SHAP 解釋：對 production XGBoost 模型算單筆貢獻度與全域 summary plot。

只用 production 模型（在整個 70% 訓練集上 fit 出來的那個，來自
train.fit_xgboost_pipeline），背景資料集是同一份訓練集，不碰測試集。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

import data as D
from train import fit_xgboost_pipeline

TOP_N_FEATURES = 5


def build_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.TreeExplainer:
    """背景資料集用訓練集，走 interventional + probability 輸出。

    model_output="probability" 讓 base_value 與 shap_value 都落在機率
    空間，兩者加總直接等於模型自己的 predict_proba（已用合成資料驗證誤
    差 <1e-5），比預設的 log-odds 空間更直接可讀，risk_score 才有意義。
    """
    cleaner = pipeline.named_steps["clean"]
    background_clean = cleaner.transform(background)
    return shap.TreeExplainer(
        pipeline.named_steps["model"],
        data=background_clean,
        feature_perturbation="interventional",
        model_output="probability",
    )


@dataclass(frozen=True)
class FeatureContribution:
    """單一特徵在這筆預測裡的貢獻。"""

    feature: str
    shap_value: float
    actual_value: float


@dataclass(frozen=True)
class InstanceExplanation:
    """單筆解釋輸出：base_value、risk_score、top-N 特徵貢獻。"""

    index: int
    base_value: float
    risk_score: float
    top_features: tuple[FeatureContribution, ...]


def explain_instance(
    pipeline: Pipeline,
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    top_n: int = TOP_N_FEATURES,
) -> InstanceExplanation:
    """對單一批次（一列 DataFrame）算 SHAP 貢獻，取 |shap_value| 最大的 top_n 個特徵。"""
    cleaner = pipeline.named_steps["clean"]
    X_clean = cleaner.transform(X_row)

    shap_values = explainer.shap_values(X_clean)[0]
    risk_score = float(pipeline.predict_proba(X_row)[0, 1])

    order = np.argsort(-np.abs(shap_values))[:top_n]
    top_features = tuple(
        FeatureContribution(
            feature=str(X_clean.columns[i]),
            shap_value=float(shap_values[i]),
            actual_value=float(X_clean.iloc[0, i]),
        )
        for i in order
    )

    return InstanceExplanation(
        index=int(X_row.index[0]),
        base_value=float(explainer.expected_value),
        risk_score=risk_score,
        top_features=top_features,
    )


def format_instance_report(explanation: InstanceExplanation) -> str:
    """把單筆解釋排成人看得懂的報告。"""
    lines = [
        f"批次 index={explanation.index}：",
        f"  base_value={explanation.base_value:.4f}  risk_score={explanation.risk_score:.4f}",
        "  top 貢獻特徵：",
    ]
    for f in explanation.top_features:
        lines.append(f"    {f.feature:>6}: shap={f.shap_value:+.4f}  actual={f.actual_value:.4f}")
    return "\n".join(lines)


def compute_shap_values(
    pipeline: Pipeline, explainer: shap.TreeExplainer, X: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """對整批資料算 SHAP 值，回傳值與清理後的特徵表（summary plot 兩者都要）。"""
    cleaner = pipeline.named_steps["clean"]
    X_clean = cleaner.transform(X)
    shap_values = explainer.shap_values(X_clean)
    return shap_values, X_clean


def save_summary_plot(shap_values: np.ndarray, X_clean: pd.DataFrame, output_path: Path) -> None:
    """存全域 SHAP summary plot（每個特徵在整個資料集上的貢獻分布）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X_clean, show=False)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    figure_path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "shap_summary.png"

    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    y_train = D.extract_labels(split.train)

    xgb_result = fit_xgboost_pipeline(X_train, y_train)
    pipeline = xgb_result.pipeline

    explainer = build_explainer(pipeline, background=X_train)

    demo_index = y_train[y_train == 1].index[0]
    explanation = explain_instance(pipeline, explainer, X_train.loc[[demo_index]])
    print("=== 單筆解釋範例（訓練集內一筆實際 Fail 批次，僅供示範）===")
    print(format_instance_report(explanation))

    print()
    print("=== 全域 SHAP summary plot（背景與計算對象皆為訓練集）===")
    shap_values, X_clean = compute_shap_values(pipeline, explainer, X_train)
    save_summary_plot(shap_values, X_clean, figure_path)
    print(f"已存至：{figure_path}")
