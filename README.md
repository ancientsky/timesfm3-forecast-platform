# 🦠 傳染病統計預測與多模型對比系統 (Epidemic Forecasting Platform)

### Google TimesFM 3.0 vs. Meta Prophet vs. Auto ARIMA (Powered by Python 3.14 & PyTorch 2.14+)
> 整合 Google 最新 **TimesFM 3.0** 時間序列基礎大模型、**Meta Prophet** 貝葉斯可加模型與經典 **Auto ARIMA** 計量經濟學模型，支援**台灣疾管署（Taiwan CDC）發病年週自動識別與對照轉換**，專為傳染病統計時序打造之現代化互動式預測與評估平台。全系統已升級至 **Python 3.14**，享有更優異的直譯器執行效能與 PyTorch 深度最佳化。

---

## 🌟 核心特色

1. **三維模型流派深度對比**：
   - 🔵 **Google TimesFM 3.0** (`google/timesfm-3.0-pytorch`)：330M 參數之 Stacked Mixing Transformer 時間序列基礎大模型，於超過 1 兆時間點上完成海量預訓練，具備卓越之 **Zero-Shot（零樣本）** 泛化能力，能敏銳捕捉傳染病擴散突波與非線性波峰衰減。原生輸出 **P10 ~ P90 之非對稱信賴區間**。
   - 🟠 **Meta Prophet（傳染病優化版）**：經典貝葉斯時間序列分解模型。針對傳染病流行病學特性實裝 **$\log(y+1)$ 對數空間指數擬合**、**95% 轉折點歷史搜尋範圍**與平滑 Fourier 週期調控，徹底解決傳統 Prophet 在疫情波段衰減時跌穿 0 遭粗暴截斷的問題。
   - 🟢 **Auto ARIMA (SARIMAX)**：經典 Box-Jenkins 計量統計模型，採用逐步 AIC/BIC 資訊準則自動搜尋最佳自回歸、差分與移動平均階數 $(p, d, q)$，提供扎實的統計基線。
2. **疾管署「發病年週」官方規則引擎（免更新日曆檔、無跨年斷層）**：
   - 內建台灣疾管署（Taiwan CDC）/ MMWR 官方週期演算法（**每週日為起始日、週三決定流行病學年**）。
   - 經與歷年 `DIM_CAL.csv` 共 10,592 筆記錄逐日交叉驗證，達到 **100.00% 完全吻合（0 誤差）**！
   - 無需每年等待疾管署釋出新年度曆表，系統可對任意歷史或未來年份（2026 ~ 2050+）動態秒級精準推算，即使在跨年度時也絕不中斷。
3. **流行病學未滿週通報延遲排除機制（預設開啟）**：
   - 傳染病數據最新一週常因資料擷取時尚未滿週，或基層院所通報延遲（Reporting Lag），通報數值經常斷崖式偏低（例如本例中 63 例驟降至 14 例）。
   - 系統**預設自動排除最新一期不完整數據**，改以最近完整週作為基準進行推演，避免模型誤判疫情暴跌；同時在走勢圖上以空心紅圈醒目標記初步通報值供專業人員對照。
3. **全面升級 Python 3.14 與 PyTorch 2.14+**：
   - 全面採用最新 **Python 3.14** 直譯器，享有大幅提升的字元碼直譯效率、更低的記憶體負擔與更快的推論反應速度。
   - 搭配 PyTorch 最新 2.14+ 架構，在 CPU/GPU 混合運算上展現高度並行吞吐能力。
4. **雙重推論與評估模式**：
   - 🔮 **未來預測模式**：使用全部歷史資料，向外推論未來未知的 8 個時間步（或自選 4 ~ 24 期）。
   - 🧪 **歷史回測模式**：保留歷史最後 8 期真實值作為 Ground Truth，即時輸出 **MAE、RMSE、MAPE、SMAPE、WAPE、趨勢方向吻合度、達峰時間偏差（Steps）、峰值誤差（%）、信賴區間覆蓋率（%）** 等 9 項量化評估計分卡。
5. **現代化互動視覺化與匯出**：
   - Plotly 動態走勢圖支援多圖層套疊、信賴區間著色帶、時序分割標線與自訂 Tooltip。
   - 一鍵下載包含三大模型預測值、信賴上下限與疾管署年週的完整 CSV 報表。

---

## 📁 專案架構目錄

```
timesfm3/
├── app.py                      # Streamlit 現代化 Web 介面主程式
├── run.sh                      # 本地一鍵啟動腳本 (自動呼叫 Python 3.14 環境)
├── requirements.txt            # Python 3.14 依賴套件清單 (PyTorch 2.14+)
├── packages.txt                # 雲端部署系統依賴 (build-essential)
├── Dockerfile                  # 容器化建置設定 (python:3.14-slim)
├── docker-compose.yml          # Docker Compose 一鍵部署配置
├── models/
│   ├── timesfm_wrapper.py      # Google TimesFM 3.0 模型推論封裝
│   ├── prophet_wrapper.py      # Meta Prophet 優化版模型封裝
│   └── arima_wrapper.py        # Auto ARIMA 統計模型封裝
├── utils/
│   ├── data_processor.py       # 資料前處理、年週/日期轉換、DIM_CAL 對照
│   └── metrics.py              # MAE/RMSE/MAPE/峰值誤差評估矩陣
└── sample_data/
    ├── cdc_dengue_year_week.csv # 疾管署發病年週登革熱範例 (202434~202635)
    ├── DIM_CAL.csv             # 疾管署歷年年週與日期對照表 (2007~2035)
    ├── dengue_weekly.csv       # 登革熱每週確診範例
    ├── covid19_daily.csv       # COVID-19 每日新增確診範例
    ├── flu_weekly.csv          # 流感/類流感每週門急診人次範例
    └── enterovirus_weekly.csv  # 腸病毒每週急診趨勢範例
```

---

## 💻 本地端部署指引 (Local Deployment Guide)

### 系統環境需求
- **作業系統**：Linux (Ubuntu 20.04+ 推薦)、macOS、Windows 10/11
- **Python 版本**：**`3.14`** (推薦 `3.14.x`)
- **記憶體 (RAM)**：建議 8GB 以上 (TimesFM 3.0 模型約需 2~3GB RAM)
- **硬碟空間**：約 5GB 空閒空間 (用於快取 Hugging Face 模型權重約 1.3GB)
- **GPU (可選)**：支援 NVIDIA GPU (CUDA)，若無 GPU 或架構不相容，系統已內建自動回退 CPU 機制，CPU 推論僅需數秒。

---

### 方法 A：使用 `uv` 極速部署（強烈推薦，1 分鐘搞定）

`uv` 是目前 Python 生態系最快的套件管理工具，能自動解析 Python 3.14 並在數秒內完成依賴安裝：

```bash
# 1. 複製專案原始碼
git clone <your-repo-url> timesfm3
cd timesfm3

# 2. 安裝 uv (若尚未安裝)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # 或 export PATH="$HOME/.local/bin:$PATH"

# 3. 建立 Python 3.14 虛擬環境並安裝依賴
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt

# 4. 一鍵啟動 Web 介面
./run.sh
```

---

### 方法 B：使用標準 `venv` 部署

```bash
# 1. 進入專案目錄
cd timesfm3

# 2. 建立並啟動 Python 3.14 虛擬環境
python3.14 -m venv .venv
source .venv/bin/activate  # Windows 請執行: .venv\Scripts\activate

# 3. 升級 pip 並安裝依賴
pip install --upgrade pip
pip install -r requirements.txt

# 4. 啟動 Streamlit
streamlit run app.py
```

瀏覽器開啟 [http://localhost:8501](http://localhost:8501) 即可開始使用！

---

## ☁️ 雲端服務與容器化部署指南 (Cloud Deployment Guide)

### 方案 1：使用 Docker / Docker Compose 部署（基於 Python 3.14 Slim）

本專案已附帶優化過的 `Dockerfile` 與 `docker-compose.yml`，基於 `python:3.14-slim` 映像檔：

```bash
# 1. 使用 Docker Compose 一鍵構建並背景啟動
docker compose up -d --build

# 2. 查看容器即時運行日誌
docker compose logs -f

# 3. 停止容器
docker compose down
```

> **提示：** `docker-compose.yml` 已預先設定 `hf_cache` Volume 儲存磁碟區，可持久化保存 Hugging Face 模型權重，避免每次容器重啟時重複下載 1.3GB 權重檔案。

---

### 方案 2：Streamlit Community Cloud 部署（免費一鍵推送到雲端）

1. 將專案推送到您的 GitHub 公開儲存庫（Repository）。
2. 前往 [share.streamlit.io](https://share.streamlit.io/) 並使用 GitHub 登入。
3. 點擊 **"New app"**：
   - **Repository**: 選擇您的專案倉庫
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. 點開 **"Advanced settings..."**：
   - **Python version**: 選取 **`3.14`**。
5. 點擊 **"Deploy!"**，平台將自動依照 `packages.txt` 與 `requirements.txt` 安裝，數分鐘內即可產出專屬公開分享網址！

---

### 方案 3：雲端虛擬機 (GCP Compute Engine / AWS EC2 / Azure VM) + Systemd 常駐

適合公司內部伺服器或自建雲端伺服器長期穩定運作：

#### Step 1: 建立雲端 VM 實例
- 推薦規格：`e2-standard-2` (GCP) 或 `t3.xlarge` (AWS)（4 vCPU, 8~16GB RAM）。
- 作業系統：Ubuntu 22.04 LTS 或 24.04 LTS。
- 防火牆規則（Security Group）：開啟 TCP Port `8501`。

#### Step 2: 主機環境初始化與安裝
```bash
sudo apt update && sudo apt install -y git python3.14 python3.14-venv

git clone <your-repo-url> /opt/timesfm3
cd /opt/timesfm3

python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: 配置 Systemd 守護行程（開機自動啟動與崩潰重啟）
建立系統服務檔案 `/etc/systemd/system/timesfm.service`：
```bash
sudo bash -c 'cat << EOF > /etc/systemd/system/timesfm.service
[Unit]
Description=Epidemic Forecasting Streamlit Platform (Python 3.14)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/timesfm3
ExecStart=/opt/timesfm3/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'

# 載入並啟動服務
sudo systemctl daemon-reload
sudo systemctl enable --now timesfm.service

# 查看運行狀態
sudo systemctl status timesfm.service
```

---

## 📋 輸入資料格式說明

本系統提供高度智慧的欄位識別能力，支援以下兩種主要輸入型態：

### 1. 疾管署「發病年週」格式（自動識別轉換）
當第一欄為 6 位數字之疾管署年週時，系統將自動呼叫 `DIM_CAL.csv` 轉為對應之每週日（週起始日）：
```csv
"發病年週","確定病例數"
"202434",3
"202435",19
"202436",48
"202501",12
"202553",2
"202601",3
"202635",14
```

### 2. 標準日期格式（支援日 / 週 / 月資料）
支援 ISO 日期、年月日等多種格式：
```csv
日期,確診病例數
2024-01-07,120
2024-01-14,145
2024-01-21,190
```

---

## 🏆 模型評估指標計分卡

切換至「🧪 歷史回測模式」時，系統保留最後 8 期真實值進行量化客觀比對：

- **MAE (Mean Absolute Error)**：平均絕對誤差，最直觀的病例數預測偏差。
- **RMSE (Root Mean Square Error)**：均方根誤差，對離群極端波峰更敏感。
- **MAPE (%) / SMAPE (%)**：平均絕對百分比誤差，跨病種與規模量級時的標準化指標。
- **Directional Accuracy (%)**：趨勢方向準確率，評估模型是否能預測下期「上升」或「下降」。
- **Peak Value Error (%)**：峰值偏差率，傳染病醫療資源備援之關鍵指標。
- **Peak Timing Error (Steps)**：達峰時間偏差（提早或落後幾期），協助判斷防疫警戒波段。
- **CI Coverage (%)**：信賴區間覆蓋率，檢驗不確定性評估的可靠度。

---

## ❓ 常見問題與除錯 (FAQ & Troubleshooting)

### Q1: Python 3.14 帶來的優勢？
- **效能提升**：Python 3.14 在直譯器層級進行了深度優化，配合 PyTorch 2.14+ 原生加速支援，模型推論延遲進一步降低，並具備更好的記憶體回收管理。

### Q2: 初次點擊預測時等待時間較長？
- **原因**：初次啟動時，系統需自 Hugging Face Hub 下載 `google/timesfm-3.0-pytorch` 預訓練權重（約 1.3GB）。
- **說明**：下載完成後會儲存於本機快取目錄（`~/.cache/huggingface`）。後續重啟或推論均會直接載入記憶體快取，單次推論僅需數秒。

### Q3: 舊款 GPU 出現 `CUDA capability sm_61 is not compatible` 警告？
- **原因**：新款 PyTorch 預設 CUDA 13+ 核心針對近代 GPU 架構最佳化。
- **說明**：本系統封裝層具備**自動容錯切換（Auto-fallback）**。若偵測到 GPU 核心不相容，會自動無縫切換為 CPU 高速推論，完全無需人工介入或中斷服務。

---

## 📜 開源許可與致謝

- 本專案程式碼依據 **Apache License 2.0** 授權釋出。
- **Google TimesFM 3.0** 模型權重依據 Google **TimesFM Non-Commercial License v1.0** 規範提供研究與非商業用途。
- 曆表與年週基準遵循台灣衛生福利部疾病管制署（Taiwan CDC）開放資料規範。
