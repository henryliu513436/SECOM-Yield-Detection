# 對齊目標環境的 Python 版本（secom conda env 用 3.12）。
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# data/raw/uci-secom.csv 與 models/metadata.json 都不進 git（前者是原
# 始資料、後者是訓練產物），映像檔本身兩者都不包含，執行時用 volume
# 掛進來：
#   docker run -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/models:/app/models" \
#     secom-api python train.py          # 先產生 models/metadata.json
#   docker run -p 8000:8000 \
#     -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/models:/app/models:ro" \
#     secom-api                          # 再啟動 API
WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
