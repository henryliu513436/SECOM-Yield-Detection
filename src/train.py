"""模型建構、交叉驗證與閾值選擇：IsolationForest（非監督）與 XGBoost（監督）。

模型建構函式照 CLAUDE.md 規定的參數把模型建起來、正確 fit；CV 部分用
TimeSeriesSplit 手寫迴圈重用同一套建構函式，讓每一折走跟單次訓練完全
相同的路徑（含早停）。閾值選擇與 confusion matrix 全部只用訓練集內部
的資料（production 模型自己的早停驗證尾端），不觸碰測試集——測試集
的最終評估是後續、待確認閾值之後才做的事。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import data as D
from features import SecomColumnCleaner

RANDOM_SEED = 42
ISOLATION_FOREST_CONTAMINATION = 0.07

XGB_MAX_DEPTH = 3
XGB_MAX_ESTIMATORS = 500
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_EVAL_METRIC = "aucpr"
EARLY_STOP_TAIL_FRACTION = 0.2
MIN_EARLY_STOP_TAIL_SIZE = 50
XGB_FALLBACK_N_ESTIMATORS = 100
XGB_THRESHOLD_MIN_PRECISION = 0.20

METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "metadata.json"
_METADATA_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")


def compute_scale_pos_weight(y: pd.Series) -> float:
    """負樣本數 / 正樣本數，用來修正 XGBoost 損失函數的類別不平衡。

    只能用實際拿去 fit 的那批標籤算，不能用全體或測試集，否則模型會
    間接看到不該看到的類別比例。
    """
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    if positive == 0:
        raise ValueError("正樣本數為 0，無法計算 scale_pos_weight")
    return negative / positive


def build_isolation_forest_pipeline(
    X: pd.DataFrame,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """在 X 上 fit 清理 + IsolationForest，不使用任何標籤。

    沒有早停這種需要中途插入驗證集的機制，一次 Pipeline.fit 就能讓兩
    步都正確地只碰這份 X。
    """
    pipeline = Pipeline(
        [
            ("clean", SecomColumnCleaner()),
            ("model", IsolationForest(contamination=contamination, random_state=random_state)),
        ]
    )
    return pipeline.fit(X)


@dataclass(frozen=True)
class XGBFitResult:
    """fit_xgboost_pipeline 的回傳值，除了模型還帶著降級資訊與早停驗證集。

    val_scores/val_labels 是早停驗證尾端的預測分數與真實標籤，用來給
    閾值選擇使用；用固定樹數的降級路徑沒有早停驗證集，兩者為 None。
    """

    pipeline: Pipeline
    used_early_stopping: bool
    n_estimators: int
    scale_pos_weight: float
    val_labels: np.ndarray | None
    val_scores: np.ndarray | None


def _fit_xgboost_with_early_stopping(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    random_state: int,
) -> XGBClassifier:
    """早停路徑：用 eval_set 決定實際樹數。

    n_jobs=1 是刻意的：預設會用滿所有核心跑 hist 演算法的直方圖建構，
    多執行緒之間浮點數加總的順序不保證每次執行都相同，即使 random_state
    固定，不同次執行仍可能選出不同的早停樹數。單執行緒才能保證每次執行
    位元級一致。
    """
    model = XGBClassifier(
        max_depth=XGB_MAX_DEPTH,
        n_estimators=XGB_MAX_ESTIMATORS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        eval_metric=XGB_EVAL_METRIC,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _fit_xgboost_fixed_estimators(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    scale_pos_weight: float,
    random_state: int,
) -> XGBClassifier:
    """降級路徑：早停尾端沒有任何 Fail，早停沒有意義，改用固定樹數。

    同樣固定 n_jobs=1，理由見 _fit_xgboost_with_early_stopping。
    """
    model = XGBClassifier(
        max_depth=XGB_MAX_DEPTH,
        n_estimators=XGB_FALLBACK_N_ESTIMATORS,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_fit, y_fit)
    return model


def fit_xgboost_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    tail_fraction: float = EARLY_STOP_TAIL_FRACTION,
    random_state: int = RANDOM_SEED,
) -> XGBFitResult:
    """在 X/y 上 fit 清理 + XGBoost，早停驗證集從尾端切一塊。

    尾端大小是 tail_fraction 與 MIN_EARLY_STOP_TAIL_SIZE 兩者取大，
    避免資料量小的折切出來的尾端小到沒有代表性。如果尾端剛好沒有任何
    Fail，代表早停這個機制在這一折沒有意義，改用固定的
    XGB_FALLBACK_N_ESTIMATORS 棵樹，不假裝早停在運作，也沒有驗證集
    可以拿來選閾值。
    """
    cleaner = SecomColumnCleaner()
    X_clean = cleaner.fit_transform(X)

    tail_size = max(int(len(X_clean) * tail_fraction), MIN_EARLY_STOP_TAIL_SIZE)
    cut = len(X_clean) - tail_size
    if cut < 1:
        raise ValueError(f"訓練資料只有 {len(X_clean)} 筆，不足以切出早停尾端")

    tail_has_fail = int(y.iloc[cut:].sum()) > 0

    if tail_has_fail:
        X_fit, X_val = X_clean.iloc[:cut], X_clean.iloc[cut:]
        y_fit, y_val = y.iloc[:cut], y.iloc[cut:]
        weight = compute_scale_pos_weight(y_fit)
        model = _fit_xgboost_with_early_stopping(X_fit, y_fit, X_val, y_val, weight, random_state)
        n_estimators_used = model.best_iteration + 1
        val_labels = y_val.to_numpy()
        val_scores = model.predict_proba(X_val)[:, 1]
    else:
        weight = compute_scale_pos_weight(y)
        model = _fit_xgboost_fixed_estimators(X_clean, y, weight, random_state)
        n_estimators_used = XGB_FALLBACK_N_ESTIMATORS
        val_labels, val_scores = None, None

    pipeline = Pipeline([("clean", cleaner), ("model", model)])
    return XGBFitResult(pipeline, tail_has_fail, n_estimators_used, weight, val_labels, val_scores)


def _report_isolation_forest(pipeline: Pipeline, X: pd.DataFrame) -> str:
    """判定的異常比例應接近設定的 contamination，用來核對模型有沒有建對。"""
    predictions = pipeline.predict(X)
    anomaly_rate = (predictions == -1).mean()
    return (
        f"IsolationForest：判定異常比例 {anomaly_rate:.2%}"
        f"（設定 contamination={ISOLATION_FOREST_CONTAMINATION:.0%}）"
    )


def _report_xgboost(result: XGBFitResult) -> str:
    """回報用的 scale_pos_weight、是否用了早停，以及實際樹數與驗證分數。"""
    if result.used_early_stopping:
        best_score = result.pipeline.named_steps["model"].best_score
        status = f"早停選出 {result.n_estimators} 棵樹，驗證集最佳 {XGB_EVAL_METRIC}={best_score:.4f}"
    else:
        status = f"未使用早停（尾端無 Fail），改用固定 {result.n_estimators} 棵樹"
    return f"XGBoost：scale_pos_weight={result.scale_pos_weight:.2f}，{status}"


@dataclass(frozen=True)
class FoldResult:
    """一折的評分結果：PR-AUC、相對基準線倍數、ROC-AUC，兩個模型都有。"""

    fold: int
    train_size: int
    val_size: int
    baseline: float
    iso_pr_auc: float
    xgb_pr_auc: float
    iso_ratio: float
    xgb_ratio: float
    iso_roc_auc: float
    xgb_roc_auc: float
    xgb_used_early_stopping: bool
    xgb_n_estimators: int


def _ratio_to_baseline(pr_auc: float, baseline: float) -> float:
    """PR-AUC 相對於該折隨機基準線的倍數。

    各折基準線差距很大（正樣本比例不同），絕對 PR-AUC 無法跨折比較，
    只有除以自己的基準線之後才能放在一起看訊號強弱。基準線為 0 或
    PR-AUC 本身是 NaN 時，倍數同樣沒有意義。
    """
    if baseline <= 0 or np.isnan(pr_auc):
        return float("nan")
    return pr_auc / baseline


def _safe_average_precision(y_true: pd.Series, scores: np.ndarray) -> float:
    """驗證折沒有正樣本時 average_precision_score 沒有意義，回傳 NaN。"""
    if int(np.sum(y_true)) == 0:
        return float("nan")
    return average_precision_score(y_true, scores)


def _safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    """ROC-AUC 需要兩個類別都出現，否則沒有意義。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, scores)


def _score_fold(
    iso_pipeline: Pipeline,
    xgb_result: XGBFitResult,
    fold_val_X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """套用兩個模型到驗證折，回傳原始分數，不判斷分數是否有意義。"""
    iso_scores = -iso_pipeline.decision_function(fold_val_X)
    xgb_scores = xgb_result.pipeline.predict_proba(fold_val_X)[:, 1]
    return iso_scores, xgb_scores


def run_cv(X: pd.DataFrame, y: pd.Series, n_splits: int) -> list[FoldResult]:
    """對 X/y 跑 TimeSeriesSplit(n_splits)，兩個模型都重用階段3的建構函式。

    每折獨立呼叫 build_isolation_forest_pipeline / fit_xgboost_pipeline，
    只傳入該折的訓練部分；驗證折（fold_val）只用來評分，絕對不會被傳進
    任何 fit。訓練折 Fail=0 時兩個模型一起跳過該折，所有降級一律印出
    來，不靜默處理。
    """
    results: list[FoldResult] = []
    for fold, (tr_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        fold_train_X, fold_train_y = X.iloc[tr_idx], y.iloc[tr_idx]
        fold_val_X, fold_val_y = X.iloc[val_idx], y.iloc[val_idx]

        if int(fold_train_y.sum()) == 0:
            print(f"  折{fold}：訓練集 Fail=0，跳過該折")
            continue

        iso_pipeline = build_isolation_forest_pipeline(fold_train_X)
        xgb_result = fit_xgboost_pipeline(fold_train_X, fold_train_y)
        if not xgb_result.used_early_stopping:
            print(f"  折{fold}：早停尾端 Fail=0，未使用早停，改用固定 {xgb_result.n_estimators} 棵樹")
        if int(fold_val_y.sum()) == 0:
            print(f"  折{fold}：驗證折 Fail=0，PR-AUC/ROC-AUC 無意義，記為 NaN")

        iso_scores, xgb_scores = _score_fold(iso_pipeline, xgb_result, fold_val_X)
        baseline = float(fold_val_y.mean())
        iso_pr_auc = _safe_average_precision(fold_val_y, iso_scores)
        xgb_pr_auc = _safe_average_precision(fold_val_y, xgb_scores)

        results.append(
            FoldResult(
                fold=fold,
                train_size=len(tr_idx),
                val_size=len(val_idx),
                baseline=baseline,
                iso_pr_auc=iso_pr_auc,
                xgb_pr_auc=xgb_pr_auc,
                iso_ratio=_ratio_to_baseline(iso_pr_auc, baseline),
                xgb_ratio=_ratio_to_baseline(xgb_pr_auc, baseline),
                iso_roc_auc=_safe_roc_auc(fold_val_y, iso_scores),
                xgb_roc_auc=_safe_roc_auc(fold_val_y, xgb_scores),
                xgb_used_early_stopping=xgb_result.used_early_stopping,
                xgb_n_estimators=xgb_result.n_estimators,
            )
        )

    return results


def _format_summary(label: str, values: list[float], n_total: int) -> str:
    """跨折 mean ± std，NaN 折不計入；順便標出有效折數，讓人知道平均是幾折算出來的。"""
    arr = np.array(values)
    n_valid = int(np.sum(~np.isnan(arr)))
    return f"  {label}：{np.nanmean(arr):.3f} ± {np.nanstd(arr):.3f}（{n_valid}/{n_total} 折有效）"


def format_cv_report(results: list[FoldResult], n_splits: int) -> str:
    """逐折結果表格（含相對基準線倍數），附上跨折 mean ± std。"""
    lines = [f"TimeSeriesSplit(n_splits={n_splits})："]
    for r in results:
        flag = "" if r.xgb_used_early_stopping else "（未早停）"
        lines.append(
            f"  折{r.fold}: train={r.train_size:>4} val={r.val_size:>4} "
            f"baseline={r.baseline:.3f}  "
            f"IsoForest={r.iso_pr_auc:.3f}({r.iso_ratio:.2f}x)  "
            f"XGB={r.xgb_pr_auc:.3f}({r.xgb_ratio:.2f}x){flag}"
        )

    n = len(results)
    lines.append(_format_summary("IsoForest PR-AUC ", [r.iso_pr_auc for r in results], n))
    lines.append(_format_summary("XGBoost   PR-AUC ", [r.xgb_pr_auc for r in results], n))
    lines.append(_format_summary("IsoForest 倍數    ", [r.iso_ratio for r in results], n))
    lines.append(_format_summary("XGBoost   倍數    ", [r.xgb_ratio for r in results], n))
    lines.append(_format_summary("IsoForest ROC-AUC", [r.iso_roc_auc for r in results], n))
    lines.append(_format_summary("XGBoost   ROC-AUC", [r.xgb_roc_auc for r in results], n))
    return "\n".join(lines)


def select_threshold_at_min_precision(
    y_true: np.ndarray, scores: np.ndarray, min_precision: float
) -> float:
    """在 precision >= min_precision 的門檻下，挑能讓 recall 最大的閾值。

    precision_recall_curve 回傳的 precision/recall 比 thresholds 多一個
    點（對應閾值=+inf、precision=1、recall=0），要先對齊長度才能配對。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]

    eligible = precision >= min_precision
    if not eligible.any():
        raise ValueError(f"沒有任何閾值能讓 precision >= {min_precision:.0%}")

    best_idx = int(np.argmax(np.where(eligible, recall, -1)))
    return float(thresholds[best_idx])


def _confusion_report(y_true: np.ndarray, predicted: np.ndarray, label: str) -> str:
    """把 confusion matrix 與 precision/recall 排成一行文字。"""
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return f"{label}：TP={tp} FP={fp} FN={fn} TN={tn}，precision={precision:.3f}，recall={recall:.3f}"


def format_threshold_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> str:
    """在給定閾值下的 XGBoost confusion matrix，用來檢視閾值實際的操作點。

    這裡的資料是 production 模型自己的早停驗證尾端，不是測試集。
    """
    predicted = (scores >= threshold).astype(int)
    return _confusion_report(y_true, predicted, f"XGBoost（閾值={threshold:.4f}）")


def format_isolation_forest_confusion(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> str:
    """IsolationForest 的判定已經由 contamination 隱含決定，不需要另外選閾值。"""
    predicted = (pipeline.predict(X) == -1).astype(int)
    return _confusion_report(y.to_numpy(), predicted, "IsolationForest（原生 contamination 判定）")


def _package_versions() -> dict[str, str]:
    """記錄產生這份 metadata 當下的套件版本。

    同樣的 random_state 在不同的 numpy/xgboost/scikit-learn 版本下會算
    出不同的閾值（實測發生過，見 README），所以 metadata 必須誠實記下
    當時的環境，而不是假裝閾值是一個跟環境無關的常數。
    """
    versions = {name: pkg_version(name) for name in _METADATA_PACKAGES}
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return versions


def save_production_metadata(
    X_train: pd.DataFrame,
    xgb_result: XGBFitResult,
    threshold: float,
    output_path: Path = METADATA_PATH,
) -> None:
    """訓練結束時輸出服務層需要的資訊，api.py 啟動時只讀這份檔案。

    閾值是訓練產物、不是程式碼常數——重訓一次，這裡的數字就可能跟著
    環境或資料變動，api.py 不應該把它寫死。
    """
    metadata = {
        "threshold": threshold,
        "feature_names": list(X_train.columns),
        "n_features": len(X_train.columns),
        "scale_pos_weight": xgb_result.scale_pos_weight,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "package_versions": _package_versions(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"

    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    y_train = D.extract_labels(split.train)

    iso_pipeline = build_isolation_forest_pipeline(X_train)
    xgb_result = fit_xgboost_pipeline(X_train, y_train)

    print(f"訓練集：{len(X_train)} 筆，Fail={int(y_train.sum())}")
    print()
    print(_report_isolation_forest(iso_pipeline, X_train))
    print(_report_xgboost(xgb_result))

    for n_splits in (5, 4):
        print()
        cv_results = run_cv(X_train, y_train, n_splits=n_splits)
        print(format_cv_report(cv_results, n_splits))

    print()
    print("=== 閾值選擇（用 production 模型自己的早停驗證尾端，不碰測試集）===")
    if xgb_result.val_scores is None:
        print("這次訓練沒有可用的早停驗證集（尾端無 Fail），無法選閾值。")
        print(f"{METADATA_PATH} 不會被寫入——沒有閾值就無法產生完整的服務層 metadata。")
    else:
        threshold = select_threshold_at_min_precision(
            xgb_result.val_labels, xgb_result.val_scores, XGB_THRESHOLD_MIN_PRECISION
        )
        print(f"precision >= {XGB_THRESHOLD_MIN_PRECISION:.0%} 條件下選出的閾值：{threshold:.4f}")
        print(format_threshold_report(xgb_result.val_labels, xgb_result.val_scores, threshold))

        tail_size = len(xgb_result.val_labels)
        X_val_tail, y_val_tail = X_train.iloc[-tail_size:], y_train.iloc[-tail_size:]
        print(format_isolation_forest_confusion(iso_pipeline, X_val_tail, y_val_tail))

        save_production_metadata(X_train, xgb_result, threshold)
        print()
        print(f"已寫入 {METADATA_PATH}（threshold={threshold:.4f}）")
