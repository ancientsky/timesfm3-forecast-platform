# 基礎映像檔：使用官方輕量 Python 3.11
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統層級必要編譯套件（用於 prophet 與 pmdarima 編譯）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴清單
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製專案原始碼
COPY . .

# 暴露 Streamlit 預設連接埠
EXPOSE 8501

# 設定 Streamlit 環境變數
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# 啟動應用程式
CMD ["streamlit", "run", "app.py"]
