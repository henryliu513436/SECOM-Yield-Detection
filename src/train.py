"""模型建構、交叉驗證與閾值選擇：IsolationForest（非監督）與 XGBoost（監督）。

模型建構函式照專案規格的參數把模型建起來、正確 fit；CV 部分用
TimeSeriesSplit 手寫迴圈重用同一套建構函式，讓每一折走跟單次訓練完全
相同的路徑。XGBoost 的 learning_rate × n_estimators 由 CV 從事先宣告
的小網格選出；閾值選擇與 confusion matrix 只用 production 模型自己的
閾值尾端（訓練集最後 20%，不參與 fit），不觸碰測試集。

為什麼不用早停：舊版用同一段尾端（219 筆、只有 14 個 Fail）做早停，
aucpr 在這麼少的正樣本上雜訊極大，production 模型在第 5 棵樹就停了，
CV ROC-AUC 只有 0.535。改成由 CV 決定樹數後，尾端也不再「先決定樹數、
又拿來選閾值」被用兩次。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import joblib
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

# data.py/features.py 用平行匯入、不是套件相對匯入，這行讓本檔案不論從
# 專案根目錄（python src/train.py、uvicorn）或從 src/ 本身執行都找得到
# 它們。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D
from features import SecomColumnCleaner

RANDOM_SEED = 42
ISOLATION_FOREST_CONTAMINATION = 0.07

XGB_MAX_DEPTH = 3
THRESHOLD_TAIL_FRACTION = 0.2
MIN_THRESHOLD_TAIL_SIZE = 50
XGB_THRESHOLD_MIN_PRECISION = 0.20
SELECTION_N_SPLITS = 5

METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "metadata.json"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
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
class XGBParams:
    """CV 網格裡的一組超參數。max_depth 固定為 XGB_MAX_DEPTH，不進網格。"""

    learning_rate: float
    n_estimators: int


# 網格在看到任何 CV 結果之前就宣告好，刻意保持很小（6 組）：組數愈多，
# 「挑出來的那組 CV 分數」被選擇偏差灌水得愈嚴重。
XGB_PARAM_GRID: tuple[XGBParams, ...] = tuple(
    XGBParams(lr, n) for lr in (0.03, 0.1) for n in (100, 300, 600)
)


@dataclass(frozen=True)
class XGBFitResult:
    """fit_xgboost_pipeline 的回傳值，除了模型還帶著閾值尾端的預測。

    val_scores/val_labels 是閾值尾端（不參與 fit）的預測分數與真實標籤，
    用來給閾值選擇使用；尾端剛好沒有 Fail 時無法選閾值，兩者為 None。
    """

    pipeline: Pipeline
    params: XGBParams
    scale_pos_weight: float
    val_labels: np.ndarray | None
    val_scores: np.ndarray | None


def _fit_xgboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    params: XGBParams,
    scale_pos_weight: float,
    random_state: int,
) -> XGBClassifier:
    """固定樹數訓練，不早停。

    n_jobs=1 是刻意的：預設會用滿所有核心跑 hist 演算法的直方圖建構，
    多執行緒之間浮點數加總的順序不保證每次執行都相同，單執行緒才能保證
    每次執行位元級一致。
    """
    model = XGBClassifier(
        max_depth=XGB_MAX_DEPTH,
        learning_rate=params.learning_rate,
        n_estimators=params.n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_fit, y_fit)
    return model


def fit_xgboost_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    params: XGBParams,
    tail_fraction: float = THRESHOLD_TAIL_FRACTION,
    random_state: int = RANDOM_SEED,
) -> XGBFitResult:
    """在 X/y 上 fit 清理 + XGBoost，尾端切一塊只拿來選閾值。

    尾端大小是 tail_fraction 與 MIN_THRESHOLD_TAIL_SIZE 兩者取大。模型只
    在尾端之前的資料上 fit，尾端的預測分數對這個模型來說是完全沒看過的
    資料，且跟部署時同一個模型、同一個分數尺度。
    """
    cleaner = SecomColumnCleaner()
    X_clean = cleaner.fit_transform(X)

    tail_size = max(int(len(X_clean) * tail_fraction), MIN_THRESHOLD_TAIL_SIZE)
    cut = len(X_clean) - tail_size
    if cut < 1:
        raise ValueError(f"訓練資料只有 {len(X_clean)} 筆，不足以切出閾值尾端")

    X_fit, X_val = X_clean.iloc[:cut], X_clean.iloc[cut:]
    y_fit, y_val = y.iloc[:cut], y.iloc[cut:]
    weight = compute_scale_pos_weight(y_fit)
    model = _fit_xgboost(X_fit, y_fit, params, weight, random_state)

    if int(y_val.sum()) > 0:
        val_labels = y_val.to_numpy()
        val_scores = model.predict_proba(X_val)[:, 1]
    else:
        val_labels, val_scores = None, None

    pipeline = Pipeline([("clean", cleaner), ("model", model)])
    return XGBFitResult(pipeline, params, weight, val_labels, val_scores)


def _report_isolation_forest(pipeline: Pipeline, X: pd.DataFrame) -> str:
    """判定的異常比例應接近設定的 contamination，用來核對模型有沒有建對。"""
    predictions = pipeline.predict(X)
    anomaly_rate = (predictions == -1).mean()
    return (
        f"IsolationForest：判定異常比例 {anomaly_rate:.2%}"
        f"（設定 contamination={ISOLATION_FOREST_CONTAMINATION:.0%}）"
    )


def _report_xgboost(result: XGBFitResult) -> str:
    """回報用的 scale_pos_weight 與實際使用的超參數。"""
    p = result.params
    return (
        f"XGBoost：scale_pos_weight={result.scale_pos_weight:.2f}，"
        f"learning_rate={p.learning_rate}，n_estimators={p.n_estimators}"
    )


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


def run_cv(X: pd.DataFrame, y: pd.Series, n_splits: int, params: XGBParams) -> list[FoldResult]:
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
        xgb_result = fit_xgboost_pipeline(fold_train_X, fold_train_y, params)
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
            )
        )

    return results


@dataclass(frozen=True)
class ParamSelectionRow:
    """網格中一組超參數的 CV 結果。"""

    params: XGBParams
    mean_pr_auc_ratio: float
    mean_roc_auc: float


def select_xgb_params(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = SELECTION_N_SPLITS,
    grid: tuple[XGBParams, ...] = XGB_PARAM_GRID,
) -> tuple[XGBParams, list[ParamSelectionRow]]:
    """在訓練集上用時間序列 CV 評估網格，回傳「PR-AUC 相對基準線倍數」
    平均最高的那組，以及完整的比較表。

    主指標用倍數而不是絕對 PR-AUC，理由跟 format_cv_report 一樣：各折
    基準線差距很大，絕對值無法跨折平均。平手時取網格裡較前面的一組。
    注意被選中那組的 CV 分數帶有選擇偏差，會略為樂觀。
    """
    table: list[ParamSelectionRow] = []
    for params in grid:
        results = run_cv(X, y, n_splits, params)
        table.append(
            ParamSelectionRow(
                params=params,
                mean_pr_auc_ratio=float(np.nanmean([r.xgb_ratio for r in results])),
                mean_roc_auc=float(np.nanmean([r.xgb_roc_auc for r in results])),
            )
        )
    best = max(table, key=lambda row: row.mean_pr_auc_ratio)
    return best.params, table


def format_selection_report(table: list[ParamSelectionRow], best: XGBParams) -> str:
    lines = ["learning_rate  n_estimators  PR-AUC倍數  ROC-AUC"]
    for row in table:
        mark = "  ← 選用" if row.params == best else ""
        lines.append(
            f"  {row.params.learning_rate:<12} {row.params.n_estimators:<13} "
            f"{row.mean_pr_auc_ratio:.2f}x      {row.mean_roc_auc:.3f}{mark}"
        )
    return "\n".join(lines)


def _format_summary(label: str, values: list[float], n_total: int) -> str:
    """跨折 mean ± std，NaN 折不計入；順便標出有效折數，讓人知道平均是幾折算出來的。"""
    arr = np.array(values)
    n_valid = int(np.sum(~np.isnan(arr)))
    return f"  {label}：{np.nanmean(arr):.3f} ± {np.nanstd(arr):.3f}（{n_valid}/{n_total} 折有效）"


def format_cv_report(results: list[FoldResult], n_splits: int) -> str:
    """逐折結果表格（含相對基準線倍數），附上跨折 mean ± std。"""
    lines = [f"TimeSeriesSplit(n_splits={n_splits})："]
    for r in results:
        lines.append(
            f"  折{r.fold}: train={r.train_size:>4} val={r.val_size:>4} "
            f"baseline={r.baseline:.3f}  "
            f"IsoForest={r.iso_pr_auc:.3f}({r.iso_ratio:.2f}x)  "
            f"XGB={r.xgb_pr_auc:.3f}({r.xgb_ratio:.2f}x)"
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

    這裡的資料是 production 模型自己的閾值尾端，不是測試集。
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


def _git_commit_hash() -> str | None:
    """目前 HEAD 的 commit hash，讓 api.py 能偵測部署的 model.pkl 是不是
    用目前這份原始碼訓練出來的。

    抓不到（例如還沒 commit、不是 git repo）就回傳 None，不能讓這個
    附加資訊的缺失擋掉整個訓練流程。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def summarize_cv(results: list[FoldResult]) -> dict:
    """把選定超參數的 CV 結果濃縮成幾個數字，寫進 metadata 供下游（例如
    agent 工具）誠實回報模型可信度，不在別處寫死。"""
    return {
        "n_splits": len(results),
        "xgb_pr_auc_ratio_mean": float(np.nanmean([r.xgb_ratio for r in results])),
        "xgb_roc_auc_mean": float(np.nanmean([r.xgb_roc_auc for r in results])),
        "xgb_roc_auc_std": float(np.nanstd([r.xgb_roc_auc for r in results])),
    }


def save_production_metadata(
    X_train: pd.DataFrame,
    xgb_result: XGBFitResult,
    threshold: float,
    cv_summary: dict,
    output_path: Path = METADATA_PATH,
) -> None:
    """訓練結束時輸出服務層需要的資訊，api.py 啟動時只讀這份檔案。

    閾值是訓練產物、不是程式碼常數——重訓一次，這裡的數字就可能跟著
    環境或資料變動，api.py 不應該把它寫死。`git_commit` 讓 api.py 能
    比對「載入的 model.pkl 是不是用目前這份原始碼訓練出來的」。
    """
    metadata = {
        "threshold": threshold,
        "feature_names": list(X_train.columns),
        "n_features": len(X_train.columns),
        "scale_pos_weight": xgb_result.scale_pos_weight,
        "xgb_params": asdict(xgb_result.params),
        "cv_summary": cv_summary,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "package_versions": _package_versions(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def save_production_model(pipeline: Pipeline, explainer, output_path: Path = MODEL_PATH) -> None:
    """把 production pipeline 跟 SHAP explainer 一起存檔，api.py 啟動時
    直接載入，不重新訓練。

    explainer 也要一起存，是因為它的背景資料集（訓練集）在部署環境不
    一定存在——production 環境不會有訓練資料，部署的模型必須就是被
    評估過的那一個，不能靠重新訓練來湊出一個「應該一樣」的模型。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "explainer": explainer}, output_path)


def load_production_params(path: Path = METADATA_PATH) -> XGBParams:
    """讀出 production 模型實際使用的超參數，給需要重建同一個模型的腳本
    （explain.py、plot_presentation_charts.py）使用，不在各處重複選參。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，請先執行 `python src/train.py`")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "xgb_params" not in metadata:
        raise ValueError(f"{path} 是舊版 metadata（沒有 xgb_params），請重新執行 `python src/train.py`")
    return XGBParams(**metadata["xgb_params"])


@dataclass(frozen=True)
class ProductionTraining:
    """train_production 的完整產物，main 與 tests/conftest.py 共用。"""

    xgb_result: XGBFitResult
    selection_table: list[ParamSelectionRow]
    cv_results: list[FoldResult]
    threshold: float | None


def train_production(X_train: pd.DataFrame, y_train: pd.Series) -> ProductionTraining:
    """選參 → 用選中的參數 fit production 模型 → 在閾值尾端選閾值。

    尾端沒有 Fail 時 threshold 為 None，由呼叫端決定怎麼處理（main 會拒
    絕寫出 metadata），不在這裡偷偷補一個預設值。
    """
    best, table = select_xgb_params(X_train, y_train)
    xgb_result = fit_xgboost_pipeline(X_train, y_train, best)
    cv_results = run_cv(X_train, y_train, SELECTION_N_SPLITS, best)

    threshold = None
    if xgb_result.val_scores is not None:
        threshold = select_threshold_at_min_precision(
            xgb_result.val_labels, xgb_result.val_scores, XGB_THRESHOLD_MIN_PRECISION
        )
    return ProductionTraining(xgb_result, table, cv_results, threshold)


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"

    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    y_train = D.extract_labels(split.train)

    print(f"訓練集：{len(X_train)} 筆，Fail={int(y_train.sum())}")
    print()
    print(f"=== XGBoost 超參數選擇（TimeSeriesSplit({SELECTION_N_SPLITS})，只用訓練集）===")
    training = train_production(X_train, y_train)
    xgb_result = training.xgb_result
    print(format_selection_report(training.selection_table, xgb_result.params))

    iso_pipeline = build_isolation_forest_pipeline(X_train)
    print()
    print(_report_isolation_forest(iso_pipeline, X_train))
    print(_report_xgboost(xgb_result))

    print()
    print(format_cv_report(training.cv_results, SELECTION_N_SPLITS))
    print()
    print(format_cv_report(run_cv(X_train, y_train, 4, xgb_result.params), 4))

    print()
    print("=== 閾值選擇（用 production 模型自己的閾值尾端，不碰測試集）===")
    if training.threshold is None:
        print("閾值尾端沒有任何 Fail，無法選閾值。")
        print(f"{METADATA_PATH} 不會被寫入——沒有閾值就無法產生完整的服務層 metadata。")
    else:
        threshold = training.threshold
        print(f"precision >= {XGB_THRESHOLD_MIN_PRECISION:.0%} 條件下選出的閾值：{threshold:.4f}")
        print(format_threshold_report(xgb_result.val_labels, xgb_result.val_scores, threshold))

        tail_size = len(xgb_result.val_labels)
        X_val_tail, y_val_tail = X_train.iloc[-tail_size:], y_train.iloc[-tail_size:]
        print(format_isolation_forest_confusion(iso_pipeline, X_val_tail, y_val_tail))

        save_production_metadata(X_train, xgb_result, threshold, summarize_cv(training.cv_results))
        print()
        print(f"已寫入 {METADATA_PATH}（threshold={threshold:.4f}）")

        from explain import build_explainer  # 延遲匯入，避免跟 explain.py 循環匯入

        explainer = build_explainer(xgb_result.pipeline, background=X_train)
        save_production_model(xgb_result.pipeline, explainer)
        print(f"已寫入 {MODEL_PATH}")
