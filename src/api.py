"""FastAPI 服務：健康檢查、預測、SHAP 解釋三個端點。

服務層不寫死任何訓練產物數字，也不在啟動時重新訓練——production 環境
預期沒有 data/raw/uci-secom.csv，部署的模型必須就是 `python train.py`
訓練、CV 評估過的那一個，不能靠「重新跑一次應該長得差不多」的模型代
替。啟動時直接載入 models/model.pkl（pipeline + SHAP explainer）與
models/metadata.json（閾值、特徵清單...），並比對 metadata 裡記錄的
git commit 是不是目前這份原始碼；不一致只印警告、不擋啟動，因為多數
程式碼修改（例如改文件字串）不影響模型本身，要不要重訓是人的判斷。
"""

import json
import logging
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# explain.py 用平行匯入、不是套件相對匯入，這行讓本檔案不論從專案根
# 目錄（uvicorn src.api:app）或從 src/ 本身執行都找得到它。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explain import explain_instance

logger = logging.getLogger("secom_api")
logging.basicConfig(level=logging.INFO)

METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "metadata.json"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"


def load_metadata(path: Path = METADATA_PATH) -> dict:
    """在模組匯入時就讀 metadata——這一步很輕量，不用等到 lifespan 才做。

    找不到檔案就直接讓服務啟動失敗，不要用預設值假裝有模型：這代表還
    沒跑過 `python train.py`，不是可以悄悄略過的情況。
    """
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，請先執行 `python train.py` 產生它")
    return json.loads(path.read_text(encoding="utf-8"))


_METADATA = load_metadata()
N_SENSOR_FEATURES: int = _METADATA["n_features"]
DECISION_THRESHOLD: float = _METADATA["threshold"]
FEATURE_NAMES: tuple[str, ...] = tuple(_METADATA["feature_names"])


def _current_git_commit() -> str | None:
    """跟 train.py 的 _git_commit_hash 同一套邏輯，用來跟 metadata 記錄的版本比對。"""
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


def warn_if_stale_model(metadata: dict) -> None:
    """model.pkl 是訓練當下的產物，原始碼後來可能改了但沒有重新訓練。

    只印警告、不擋啟動：多數修改不影響模型本身，擋啟動反而妨礙開發，
    要不要重訓是人的判斷，不是這支程式該擅自決定的事。兩邊有任一個
    抓不到 commit（例如 metadata 是舊版產生的、或這裡不是 git repo）
    就跳過比對，不誤報。
    """
    trained_commit = metadata.get("git_commit")
    current_commit = _current_git_commit()
    if trained_commit is None or current_commit is None:
        return
    if trained_commit != current_commit:
        logger.warning(
            "models/model.pkl 是在 commit %s 訓練的，目前程式碼是 %s，"
            "模型可能已經過期，建議重新執行 `python train.py`。",
            trained_commit[:12],
            current_commit[:12],
        )


class ModelState:
    """啟動時載入、請求時唯讀重用的模型與 SHAP explainer。"""

    def __init__(self, pipeline, explainer) -> None:
        self.pipeline = pipeline
        self.explainer = explainer


def load_model_state(path: Path = MODEL_PATH) -> ModelState:
    """從 model.pkl 載入 production pipeline 與 SHAP explainer。

    不重新訓練——見本檔案開頭的說明。
    """
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，請先執行 `python train.py` 產生它")
    bundle = joblib.load(path)
    return ModelState(bundle["pipeline"], bundle["explainer"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    warn_if_stale_model(_METADATA)
    logger.info("啟動中：載入 %s...", MODEL_PATH)
    app.state.model = load_model_state()
    logger.info("模型就緒。")
    yield


app = FastAPI(title="SECOM 良率異常偵測 API", lifespan=lifespan)


class PredictRequest(BaseModel):
    lot_id: Optional[str] = Field(default=None, description="批次識別碼，僅用於記錄 log")
    features: List[Optional[float]] = Field(
        ..., description=f"{N_SENSOR_FEATURES} 個感測器讀值，依訓練資料的欄位順序排列，缺失值可傳 null"
    )

    @field_validator("features")
    @classmethod
    def check_length(cls, value: List[Optional[float]]) -> List[Optional[float]]:
        if len(value) != N_SENSOR_FEATURES:
            raise ValueError(f"features 長度必須是 {N_SENSOR_FEATURES}，實際收到 {len(value)}")
        return value


class PredictResponse(BaseModel):
    lot_id: Optional[str]
    risk_score: float
    is_fail_predicted: bool
    threshold: float


class FeatureContributionResponse(BaseModel):
    feature: str
    shap_value: float
    actual_value: float


class ExplainResponse(BaseModel):
    lot_id: Optional[str]
    base_value: float
    risk_score: float
    top_features: List[FeatureContributionResponse]


def _request_to_frame(request: PredictRequest) -> pd.DataFrame:
    """把請求裡的原始感測器欄組成一列 DataFrame，欄名對齊 metadata 記錄的訓練欄位順序。

    只有一列時，若某欄剛好是 null，pandas 會把該欄整欄推斷成 object
    dtype（不是 float64+NaN），fillna 之後仍是 object，XGBoost 會直接
    拒絕預測。實測會在真的有缺失值的請求上炸掉，`.astype(float)` 強制
    轉型才能保證欄位一定是數值型別。
    """
    frame = pd.DataFrame([request.features], columns=list(FEATURE_NAMES))
    return frame.astype(float)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """風險分數＝未校準的 predict_proba，只能當排序分數，見 README 說明。"""
    start = time.perf_counter()
    model: ModelState = app.state.model
    X_row = _request_to_frame(request)
    risk_score = float(model.pipeline.predict_proba(X_row)[0, 1])
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict lot_id=%s risk_score=%.4f elapsed_ms=%.1f", request.lot_id, risk_score, elapsed_ms
    )
    return PredictResponse(
        lot_id=request.lot_id,
        risk_score=risk_score,
        is_fail_predicted=risk_score >= DECISION_THRESHOLD,
        threshold=DECISION_THRESHOLD,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictRequest) -> ExplainResponse:
    start = time.perf_counter()
    model: ModelState = app.state.model
    X_row = _request_to_frame(request)
    explanation = explain_instance(model.pipeline, model.explainer, X_row)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "explain lot_id=%s risk_score=%.4f elapsed_ms=%.1f",
        request.lot_id,
        explanation.risk_score,
        elapsed_ms,
    )
    return ExplainResponse(
        lot_id=request.lot_id,
        base_value=explanation.base_value,
        risk_score=explanation.risk_score,
        top_features=[
            FeatureContributionResponse(
                feature=f.feature, shap_value=f.shap_value, actual_value=f.actual_value
            )
            for f in explanation.top_features
        ],
    )
