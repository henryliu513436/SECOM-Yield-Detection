"""測試集最終評估：套用已訓練好的 production 模型在 471 筆測試集上，算出
真正的良率偵測表現。

這是全流程唯一允許真正「使用」測試集的地方——保留欄位、補值中位數、
決策閾值全部已經在訓練階段從訓練集凍結好（見 features.py、train.py），
這裡純粹套用 models/model.pkl，不重新訓練 XGBoost、不調整任何參數、
不回頭改閾值。IsolationForest 從未進入 production（train.py 只把
XGBoost pipeline 存進 model.pkl），這裡重新 fit 一份只是為了跟 CV 報告
維持同樣的兩模型並列對照，不影響任何已訓練好的產物。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# data.py/train.py 用平行匯入，這行讓本檔案不論從專案根目錄或從 scripts/
# 本身執行都找得到 src/ 底下的模組。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as D
from train import (
    _ratio_to_baseline,
    _safe_average_precision,
    _safe_roc_auc,
    build_isolation_forest_pipeline,
    format_isolation_forest_confusion,
    format_threshold_report,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_production_bundle() -> tuple[object, float]:
    """讀 model.pkl 跟 metadata.json，回傳 (XGBoost pipeline, 已凍結的決策閾值)。

    不重新訓練——閾值是訓練當下的產物，這裡只能讀，不能重算。
    """
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    bundle = joblib.load(MODELS_DIR / "model.pkl")
    return bundle["pipeline"], metadata["threshold"]


def evaluate_xgboost(pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float, baseline: float) -> None:
    scores = pipeline.predict_proba(X_test)[:, 1]
    print("=== XGBoost（production 模型，閾值取自 metadata.json，未重新訓練）===")
    print(format_threshold_report(y_test.to_numpy(), scores, threshold))
    pr_auc = _safe_average_precision(y_test, scores)
    roc_auc = _safe_roc_auc(y_test, scores)
    print(f"PR-AUC={pr_auc:.3f}（{_ratio_to_baseline(pr_auc, baseline):.2f}x baseline）  ROC-AUC={roc_auc:.3f}")


def evaluate_isolation_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, baseline: float) -> None:
    """重新 fit 在訓練集上——這個模型從未進入 production，這裡只是重建對照基準。"""
    pipeline = build_isolation_forest_pipeline(X_train)
    scores = -pipeline.decision_function(X_test)
    print("=== IsolationForest（對照基準，重新 fit 在訓練集上，從未進入 production）===")
    print(format_isolation_forest_confusion(pipeline, X_test, y_test))
    pr_auc = _safe_average_precision(y_test, scores)
    roc_auc = _safe_roc_auc(y_test, scores)
    print(f"PR-AUC={pr_auc:.3f}（{_ratio_to_baseline(pr_auc, baseline):.2f}x baseline）  ROC-AUC={roc_auc:.3f}")


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)

    X_train = D.extract_features(split.train)
    X_test = D.extract_features(split.test)
    y_test = D.extract_labels(split.test)
    baseline = float(y_test.mean())

    xgb_pipeline, threshold = load_production_bundle()

    print(f"測試集：{len(X_test)} 筆，Fail={int(y_test.sum())}（baseline={baseline:.3f}）")
    print()
    evaluate_xgboost(xgb_pipeline, X_test, y_test, threshold, baseline)
    print()
    evaluate_isolation_forest(X_train, X_test, y_test, baseline)


if __name__ == "__main__":
    main()
