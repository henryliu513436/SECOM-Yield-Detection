"""CLAUDE.md 階段6指定的 API 測試：欄位數錯誤要回 422。

刻意不用 `with TestClient(app) as client:`，讓 lifespan（訓練 production
模型）不會被觸發——欄位數驗證發生在 Pydantic 這一層，比進到 endpoint、
碰到模型更早，這樣測試才能維持又快又獨立於模型訓練。
"""

from fastapi.testclient import TestClient

from api import N_SENSOR_FEATURES, app

client = TestClient(app)


def test_api_rejects_wrong_shape() -> None:
    wrong_length = N_SENSOR_FEATURES - 1
    response = client.post("/predict", json={"features": [0.0] * wrong_length})

    assert response.status_code == 422
    assert str(N_SENSOR_FEATURES) in response.json()["detail"][0]["msg"]
