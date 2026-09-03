"""FastAPI 服務：健康檢查、預測、SHAP 解釋三個端點。

服務層不寫死任何訓練產物數字——閾值、特徵清單這些東西是訓練出來的，
不是程式碼常數，一律從 `python train.py` 產生的 models/metadata.json
讀。這樣重訓一次（甚至換一套套件版本重訓，見 README「套件版本會影響
數字」章節）之後，這支程式完全不用改。

模型本身（清理 pipeline + XGBoost）在服務啟動時 fit 一次，快取在
app.state 裡；之後每個請求重用同一個已訓練好的物件，不重新訓練、不碰
測試集。SHAP explainer 的背景資料集同樣是訓練集，一併在啟動時建好。
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

import data as D
from explain import build_explainer, explain_instance
from train import fit_xgboost_pipeline

logger = logging.getLogger("secom_api")
logging.basicConfig(level=logging.INFO)

METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "metadata.json"


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


class ModelState:
    """啟動時建立、請求時唯讀重用的模型與 SHAP explainer。"""

    def __init__(self, pipeline, explainer) -> None:
        self.pipeline = pipeline
        self.explainer = explainer


def load_model_state() -> ModelState:
    """訓練 production pipeline 並建立 SHAP explainer，服務啟動時只呼叫一次。"""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci-secom.csv"
    raw = D.load_raw(csv_path)
    split = D.split_by_time(raw)
    X_train = D.extract_features(split.train)
    y_train = D.extract_labels(split.train)

    xgb_result = fit_xgboost_pipeline(X_train, y_train)
    explainer = build_explainer(xgb_result.pipeline, background=X_train)
    return ModelState(xgb_result.pipeline, explainer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("啟動中：訓練 production 模型與 SHAP explainer...")
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
    """把請求裡的原始感測器欄組成一列 DataFrame，欄名對齊 metadata 記錄的訓練欄位順序。"""
    return pd.DataFrame([request.features], columns=list(FEATURE_NAMES))


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
