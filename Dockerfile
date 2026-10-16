# 推論服務（FastAPI）與 agent CLI 共用同一個映像檔，由 docker-compose.yml
# 用不同的 command 啟動。models/ 與 data/ 不進映像檔，執行時以 volume 掛載
# ——模型與 DB 是訓練／建置產物，不該被烤進映像檔裡。
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED=1
