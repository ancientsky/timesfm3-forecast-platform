"""
Internationalization (i18n) Module for TimesFM 3.0 vs. Prophet vs. Auto ARIMA Forecasting Platform.
Supports real-time switching between Traditional Chinese (繁體中文, 'zh') and English ('en').
"""

from typing import Dict, Any, List, Optional, Tuple

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ---------------- Page Config & Badges ----------------
    "page_title": {
        "zh": "TimesFM 3.0 vs. Prophet vs. Auto ARIMA 時序預測系統",
        "en": "TimesFM 3.0 vs. Prophet vs. Auto ARIMA Time Series Platform"
    },
    "main_header": {
        "zh": "📈 TimesFM 3.0 vs. Prophet vs. Auto ARIMA 時序預測系統",
        "en": "📈 TimesFM 3.0 vs. Prophet vs. Auto ARIMA Time Series Platform"
    },
    "badge_tfm": {
        "zh": "Google TimesFM 3.0 (Transformer 大模型)",
        "en": "Google TimesFM 3.0 (Foundation Transformer)"
    },
    "badge_prophet": {
        "zh": "Meta Prophet",
        "en": "Meta Prophet"
    },
    "badge_arima": {
        "zh": "Auto ARIMA (統計基準)",
        "en": "Auto ARIMA (Statistical Benchmark)"
    },
    "badge_engine": {
        "zh": "疾管署 EpiWeek 數學引擎 & 央行 OpenData 即時同步",
        "en": "Taiwan CDC EpiWeek Engine & Central Bank Live Sync"
    },

    # ---------------- Sidebar - Main ----------------
    "language_selector_label": {
        "zh": "🌐 語言選擇 (Language)",
        "en": "🌐 Language"
    },
    "sidebar_header": {
        "zh": "## ⚙️ 資料與預測配置",
        "en": "## ⚙️ Data & Forecast Configuration"
    },
    "data_source_label": {
        "zh": "選擇時序分析資料來源：",
        "en": "Select Time Series Data Source:"
    },

    # ---------------- Dataset Presets ----------------
    "ds_enterovirus": {
        "zh": "👶 腸病毒每週門急診就診人次 (週資料 201601~202635)",
        "en": "👶 Enterovirus Weekly Outpatient Visits (Weekly 201601~202635)"
    },
    "ds_dengue": {
        "zh": "🦟 台灣登革熱每週確診統計 (發病年週 202301~202636)",
        "en": "🦟 Taiwan Dengue Weekly Confirmed Cases (EpiWeek 202301~202636)"
    },
    "ds_scrub_typhus": {
        "zh": "🦗 恙蟲病確定病例發病月趨勢圖 (月資料 200001~202612)",
        "en": "🦗 Scrub Typhus Monthly Confirmed Cases (Monthly 200001~202612)"
    },
    "ds_covid": {
        "zh": "😷 COVID-19 每日新增本土病例 (日資料 20240901~20260904)",
        "en": "😷 COVID-19 Daily Domestic Confirmed Cases (Daily 20240901~20260904)"
    },
    "ds_flu": {
        "zh": "🌡️ 流感/類流感每週門急診就診人次 (週資料 201801~202635)",
        "en": "🌡️ Influenza/ILI Weekly Outpatient Visits (Weekly 201801~202635)"
    },
    "ds_cbc_forex": {
        "zh": "💵 銀行間市場新台幣對美元收盤匯率 (日資料 央行即時連線)",
        "en": "💵 Interbank NTD/USD Daily Closing Rate (Central Bank Live Sync)"
    },
    "ds_custom_url": {
        "zh": "🌐 外部自訂 OpenData / CSV 固定網址 (即時同步)",
        "en": "🌐 External OpenData / CSV Permanent URL (Live Sync)"
    },
    "ds_upload": {
        "zh": "📂 自訂上傳檔案 (CSV / Excel / JSON)",
        "en": "📂 Upload Custom Dataset (CSV / Excel / JSON)"
    },
    "ds_paste": {
        "zh": "✏️ 線上手動輸入 / 貼上 CSV (支援年週與日報)",
        "en": "✏️ Online Manual Input / Paste CSV (EpiWeek & Daily)"
    },

    # ---------------- CBC Live & Custom URL ----------------
    "cbc_header": {
        "zh": "##### 🏛️ 中華民國中央銀行 OpenData",
        "en": "##### 🏛️ Central Bank of the ROC (Taiwan) OpenData"
    },
    "cbc_caption": {
        "zh": "固定端點：`FTDOpenData015.csv`（2008 年至今各營業日銀行間收盤匯率）",
        "en": "Permanent Endpoint: `FTDOpenData015.csv` (Daily closing rates from 2008 to present)"
    },
    "cbc_status_connecting": {
        "zh": "正在連線央行 OpenData 端點...",
        "en": "Connecting to Central Bank OpenData endpoint..."
    },
    "cbc_status_complete": {
        "zh": "央行數據載入完畢",
        "en": "Central Bank data successfully loaded"
    },
    "cbc_status_error": {
        "zh": "連線發生異常",
        "en": "Connection error occurred"
    },
    "cbc_success_live": {
        "zh": "🟢 央行即時連線同步成功（共 {count} 筆交易日資料，至最新營業日）",
        "en": "🟢 Central Bank live sync successful ({count} trading days up to latest session)"
    },
    "cbc_fallback_info": {
        "zh": "🛡️ {msg}（共 {count} 筆）",
        "en": "🛡️ {msg} ({count} records)"
    },
    "cbc_expander_title": {
        "zh": "ℹ️ 為什麼會顯示備援資料庫？",
        "en": "ℹ️ Why is the backup database shown?"
    },
    "cbc_expander_desc": {
        "zh": (
            "**中央銀行端點防護機制說明：**\n"
            "中央銀行為國家關鍵金融基礎設施，官方伺服器設有嚴格的 WAF 防火牆與 Geo-IP 境外/雲端主機 IP 阻擋機制 (HTTP 403)。"
            "當系統處於公有雲、容器環境或代理伺服器下時，自動化請求常會被主動防護阻斷。"
            "系統內建已同步至最新營業日之高可靠性本地資料庫（至 2026-09-07 收盤價 31.552），模型推論與預測精準度 100% 不受影響！"
        ),
        "en": (
            "**Central Bank Endpoint Security Policy:**\n"
            "The Central Bank of the ROC (Taiwan) is critical national financial infrastructure. Its servers enforce strict Web Application Firewall (WAF) and Geo-IP filters (HTTP 403) blocking automated or cloud-hosted IP requests.\n"
            "When running in cloud or containerized sandbox environments, external egress may be filtered. The platform automatically activates a local high-reliability fallback dataset synced to the latest trading day (2026-09-07, close 31.552). Forecast models and accuracy remain 100% unaffected!"
        )
    },
    "cbc_resync_button": {
        "zh": "🔄 立即重新同步央行匯率",
        "en": "🔄 Re-sync Central Bank Rates Now"
    },
    "cbc_resync_help": {
        "zh": "清除快取並重新連線中央銀行伺服器",
        "en": "Clear cache and reconnect to Central Bank server"
    },
    "custom_url_header": {
        "zh": "##### 🌐 外部自訂 CSV / OpenData 固定網址",
        "en": "##### 🌐 Custom CSV / OpenData Permanent URL"
    },
    "custom_url_label": {
        "zh": "輸入資料固定下載網址 (URL)：",
        "en": "Enter Permanent Download URL (HTTP/HTTPS):"
    },
    "custom_url_help": {
        "zh": "支援政府資料開放平台 (data.gov.tw)、各機關局處公開數據或雲端 CSV 檔案連結。內建 SSRF 安全防護與多種編碼相容。",
        "en": "Supports data.gov.tw, ministry portals, GitHub raw files, or cloud CSV links. Built-in SSRF protection and encoding auto-detection."
    },
    "custom_url_status_loading": {
        "zh": "正在自遠端伺服器下載與解析時序數據...",
        "en": "Downloading and parsing time series from remote endpoint..."
    },
    "custom_url_status_complete": {
        "zh": "遠端時序數據已載入",
        "en": "Remote time series dataset loaded"
    },
    "custom_url_status_error": {
        "zh": "同步失敗",
        "en": "Sync failed"
    },
    "custom_url_success": {
        "zh": "🟢 遠端網址即時同步成功（共 {count} 筆）",
        "en": "🟢 Remote URL live sync successful ({count} rows)"
    },
    "custom_url_fallback_info": {
        "zh": "ℹ️ {msg}（共 {count} 筆）",
        "en": "ℹ️ {msg} ({count} records)"
    },
    "custom_url_resync_button": {
        "zh": "🔄 立即重新同步遠端網址",
        "en": "🔄 Re-sync Remote URL Now"
    },
    "custom_url_resync_help": {
        "zh": "清除快取並重新自遠端網址拉取最新資料",
        "en": "Clear cache and refetch latest data from remote URL"
    },
    "load_failed": {
        "zh": "數據載入失敗",
        "en": "Data loading failed"
    },
    "parse_failed": {
        "zh": "檔案解析失敗",
        "en": "File parsing failed"
    },
    "custom_url_parse_failed": {
        "zh": "網址解析失敗",
        "en": "URL parsing failed"
    },
    "upload_label": {
        "zh": "上傳時序統計檔案（支援發病年週、日/月/年日期）",
        "en": "Upload Time Series File (Supports CDC EpiWeek, Date, etc.)"
    },
    "upload_success": {
        "zh": "成功安全載入 {name}（共 {count} 筆）",
        "en": "Successfully loaded {name} safely ({count} records)"
    },
    "paste_label": {
        "zh": "貼上 CSV 格式數據（直接貼上疾管署年週統計）：",
        "en": "Paste CSV format data (CDC EpiWeek or custom series):"
    },
    "csv_parse_failed": {
        "zh": "CSV 解析失敗",
        "en": "CSV parsing failed"
    },
    "no_data_error": {
        "zh": "請在左側選取或上傳有效的時序數據。",
        "en": "Please select or upload a valid time series dataset from the sidebar."
    },

    # ---------------- Column Pickers & Preprocess ----------------
    "col_date_label": {
        "zh": "日期 / 年週欄位 (Date / Year-Week)：",
        "en": "Date / EpiWeek Column (Date / Year-Week):"
    },
    "col_target_label": {
        "zh": "時序預測目標指標 (Target Value)：",
        "en": "Forecast Target Value (Target Value):"
    },
    "time_window_label": {
        "zh": "⏳ 歷史觀測時窗 (Time Window)：",
        "en": "⏳ Historical Time Window:"
    },
    "time_window_help": {
        "zh": "若歷史早期有超大規模特異疫情或極端波動，常會導致近年細部走勢在線性圖表中被壓平。可選擇縮小觀測時窗以獲得更貼近現況的基線。",
        "en": "If early history contains extreme outbreaks or anomalies, recent subtle trends can be compressed in linear charts. Select a recent window to enhance baseline resolution."
    },
    "time_window_all": {
        "zh": "全量歷史記錄 (全部歷史數據)",
        "en": "Full Historical Series (All Available Data)"
    },
    "time_window_from": {
        "zh": "從 {year} 年起 (聚焦近年走勢)",
        "en": "From {year} Onwards (Focus on Recent Trends)"
    },

    # ---------------- Horizon & Frequency ----------------
    "forecast_params_header": {
        "zh": "### 🎯 預測參數設定",
        "en": "### 🎯 Forecast Parameters"
    },
    "horizon_label": {
        "zh": "預測期數 (Forecast Horizon Steps)：",
        "en": "Forecast Horizon Steps:"
    },
    "horizon_help": {
        "zh": "預設為未來 8 期預測（如 8 天 / 8 週 / 8 個月）",
        "en": "Default: 8 future steps (e.g. 8 days / 8 weeks / 8 months)"
    },
    "frequency_label": {
        "zh": "時間序列頻率 (Frequency)：",
        "en": "Time Series Frequency:"
    },
    "freq_weekly": {
        "zh": "週 (Weekly 'W')",
        "en": "Weekly ('W')"
    },
    "freq_daily": {
        "zh": "日 (Daily 'D')",
        "en": "Daily ('D')"
    },
    "freq_monthly": {
        "zh": "月 (Monthly 'M')",
        "en": "Monthly ('M')"
    },
    "freq_w": {
        "zh": "週資料",
        "en": "Weekly"
    },
    "freq_d": {
        "zh": "日資料",
        "en": "Daily"
    },
    "freq_m": {
        "zh": "月資料",
        "en": "Monthly"
    },
    "eval_mode_label": {
        "zh": "預測評估模式：",
        "en": "Evaluation Mode:"
    },
    "eval_mode_future": {
        "zh": "🔮 未來預測模式 (預測未知的未來 8 期)",
        "en": "🔮 Future Forecast (Predict upcoming unknown periods)"
    },
    "eval_mode_backtest": {
        "zh": "🧪 歷史回測模式 (保留最後 8 期做真實值對比評估)",
        "en": "🧪 Historical Backtest (Hold out last periods for ground-truth comparison)"
    },
    "exclude_last_label": {
        "zh": "🛡️ 預設排除最新一期不完整數據 (建議流行病週/月報開啟)",
        "en": "🛡️ Exclude Latest Incomplete Period (Recommended for surveillance data)"
    },
    "exclude_last_help": {
        "zh": "傳染病監測數據之最新一週/期常因統計尚未滿週或通報延遲（Reporting Lag）而明顯偏低。預設排除此未滿期數據，避免模型誤判疫情急速崩跌。若為日報或已結算之金融匯率數據，可取消勾選以包含最新記錄。",
        "en": "Latest reporting period in surveillance series is frequently incomplete due to reporting lag, showing artificial cliff drops. Enabling this excludes partial periods to prevent model distortion."
    },
    "confidence_interval_label": {
        "zh": "不確定性信賴區間 (Uncertainty Interval)：",
        "en": "Uncertainty Confidence Interval:"
    },
    "ci_format": {
        "zh": "{pct}% 信賴區間 (P{low} ~ P{high})",
        "en": "{pct}% CI (P{low} ~ P{high})"
    },

    # ---------------- Animation Settings ----------------
    "anim_section_header": {
        "zh": "### 🎭 等待過場動畫",
        "en": "### 🎭 Transition Animations"
    },
    "anim_theme_label": {
        "zh": "過場動畫風格：",
        "en": "Animation Theme Style:"
    },
    "anim_theme_help": {
        "zh": "在執行三模型推論等待時呈現之趣味過場動態，可隨機驚喜或指定喜愛的視覺風格！",
        "en": "Engaging animations during model inference. Choose a visual style or random surprises!"
    },
    "anim_preview_button": {
        "zh": "👁️ 立即預覽此動畫",
        "en": "👁️ Preview Animation Now"
    },
    "anim_preview_help": {
        "zh": "不用等待推論即可直接觀賞所選之過場動畫動態！",
        "en": "Preview the selected animation style instantly without waiting for inference!"
    },
    "anim_preview_close": {
        "zh": "✕ 關閉預覽",
        "en": "✕ Close Preview"
    },
    "anim_preview_banner": {
        "zh": "💡 **過場動畫即時預覽效果**（正式推論時將依據模型進度 33% ➜ 66% ➜ 100% 推進）：",
        "en": "💡 **Animation Live Preview** (During actual runs, progress advances 33% ➜ 66% ➜ 100%):"
    },
    "anim_preview_title": {
        "zh": "過場動畫即時預覽效果（正式推論時將依據模型進度 33% ➜ 66% ➜ 100% 推進）：",
        "en": "Animation Live Preview (Progress advances 33% ➜ 66% ➜ 100% during inference):"
    },
    "anim_preview_step_desc": {
        "zh": "【{theme}】動態展示中...",
        "en": "Displaying [{theme}] dynamic preview..."
    },
    "btn_close_preview": {
        "zh": "✕ 關閉預覽",
        "en": "✕ Close Preview"
    },

    # ---------------- Notices & System Explanations ----------------
    "notices_expander_title": {
        "zh": "系統資料處理與流行病學防護說明",
        "en": "System Data Processing & Epidemiological Safeguards"
    },
    "chip_ym": {
        "zh": "疾管署發病年月: {start}~{end}",
        "en": "CDC Year-Month: {start}~{end}"
    },
    "chip_yw": {
        "zh": "疾管署年週: {start}~{end}",
        "en": "CDC EpiWeek: {start}~{end}"
    },
    "chip_excluded": {
        "zh": "排除{unit}: {label} ({val}{cases})",
        "en": "Excluded {unit}: {label} ({val} {cases})"
    },
    "info_ym_title": {
        "zh": "疾管署發病年月精準對照：",
        "en": "Taiwan CDC Year-Month Precise Alignment:"
    },
    "info_ym_body": {
        "zh": "系統已自動識別欄位 **「{col}」** 為疾管署發病年月（{start} ~ {end}），共 {count} 個月。時序頻率自動校準為「月資料 (Monthly 'M')」，並以每月 1 日為標準錨定日進行時序建模。",
        "en": "System identified column **'{col}'** as Taiwan CDC Year-Month ({start} ~ {end}), spanning {count} months. Frequency calibrated to Monthly ('M') anchored to the 1st day of each month."
    },
    "info_yw_title": {
        "zh": "疾管署年週精準對照：",
        "en": "Taiwan CDC / MMWR EpiWeek Precise Alignment:"
    },
    "info_yw_body": {
        "zh": "系統已自動識別欄位 **「{col}」** 為疾管署發病年週（{start} ~ {end}），採用台灣疾管署（Taiwan CDC）/ MMWR 官方週期數學演算法（週日為每週起始日、內含週三所屬年為年週所屬年）完成轉換。本演算法直接內建跨年度數學公理，具備無限期推演能力，完全不受限於年度靜態日曆表！",
        "en": "System identified column **'{col}'** as Taiwan CDC EpiWeek ({start} ~ {end}), converted via CDC / MMWR mathematical algorithm (Sunday-Saturday, Wednesday week-1 anchor). Features perpetual year-boundary coverage without static calendar limits!"
    },
    "warn_excluded_title": {
        "zh": "流行病學{unit}防護機制生效中：",
        "en": "Epidemiological {unit} Safeguard Active:"
    },
    "warn_excluded_body": {
        "zh": "偵測到最後一期（**{label}**）初步通報病例數為 **{val} {cases}**，常為統計未滿期或通報延遲（Reporting Lag）導致的人為偏低。系統已自動將其排除，改以最近完整期別 **{last_label} ({last_val} {cases})** 為基準進行建模推估。*(若需將該期直接納入訓練，請在左側側邊欄取消勾選「排除最新一期不完整數據」)*",
        "en": "Detected latest period (**{label}**) preliminary report of **{val} {cases}**, typically artificially low due to incomplete statistical window or reporting lag. Automatically excluded; baseline anchored to latest complete period **{last_label} ({last_val} {cases})**. *(To include this period, uncheck 'Exclude Latest Incomplete Period' in the sidebar)*"
    },
    "btn_dismiss_notices": {
        "zh": "✕ 關閉此公告 (不再顯示)",
        "en": "✕ Dismiss Notice (Do not show again)"
    },
    "btn_dismiss_help": {
        "zh": "點擊後於本次操作中隱藏此說明區域，可在側邊欄隨時重新顯示",
        "en": "Hide this notice area for this session. Can be restored anytime from sidebar."
    },
    "dismiss_banner_btn": {
        "zh": "✕ 關閉此公告 (不再顯示)",
        "en": "✕ Dismiss Notice"
    },
    "dismiss_banner_help": {
        "zh": "點擊後於本次操作中隱藏此說明區域，可在側邊欄隨時重新顯示",
        "en": "Click to hide this notice during the session. Can be restored from sidebar."
    },

    # ---------------- Overview Cards & Dynamic Steps ----------------
    "ov_training_steps": { "zh": "建模使用期數", "en": "Training Steps" },
    "ov_training_steps_help": { "zh": "總觀測 {total} 期 (已排除 {ex} 期{unit})", "en": "Total observed: {total} steps ({ex} {unit} excluded)" },
    "ov_baseline_val": { "zh": "基準觀測值", "en": "Baseline Value" },
    "ov_baseline_val_help": { "zh": "最近完整期別: {label}", "en": "Latest complete period: {label}" },
    "ov_peak_val": { "zh": "歷史峰值", "en": "Historical Peak" },
    "ov_peak_val_help": { "zh": "發生於 {date}", "en": "Occurred on {date}" },
    "ov_frequency": { "zh": "時序頻率", "en": "Frequency" },
    "ov_recent_trend": { "zh": "近期趨勢 (4期變化)", "en": "Recent Trend (4-Step Change)" },
    "metric_model_steps": { "zh": "建模使用期數", "en": "Training Steps" },
    "metric_model_steps_help": { "zh": "總觀測 {total} 期 (已排除 {ex_count} 期{unit})", "en": "Total observed: {total} {periods} ({ex_count} {unit} excluded)" },
    "metric_baseline_val": { "zh": "基準觀測值", "en": "Baseline Value" },
    "metric_baseline_help": { "zh": "最近完整期別: {yw}", "en": "Latest complete period: {yw}" },
    "metric_peak_val": { "zh": "歷史峰值", "en": "Historical Peak" },
    "metric_peak_help": { "zh": "發生於 {date}", "en": "Occurred on {date}" },
    "metric_frequency": { "zh": "時序頻率", "en": "Frequency" },
    "metric_recent_trend": { "zh": "近期趨勢 (4期變化)", "en": "Recent Trend (4-Step)" },
    "unit_partial_week": { "zh": "未滿週", "en": "partial week" },
    "unit_partial_month": { "zh": "未滿月", "en": "partial month" },
    "unit_week_incomplete": { "zh": "未滿週", "en": "partial week" },
    "unit_month_incomplete": { "zh": "未滿月", "en": "partial month" },
    "unit_cases": { "zh": "例", "en": "cases" },
    "unit_counts": { "zh": "次/例", "en": "visits/cases" },
    "unit_seconds": { "zh": "秒", "en": "s" },
    "unit_periods": { "zh": "期", "en": "steps" },
    "unit_year": { "zh": "年", "en": "Year" },
    "unit_month": { "zh": "月", "en": "Month" },
    "unit_week": { "zh": "週", "en": "Week" },
    "unit_day": { "zh": "日", "en": "Day" },
    "colon": { "zh": "：", "en": ": " },
    "range_months": { "zh": "1 ~ 12 月", "en": "Jan ~ Dec (1~12 Months)" },
    "range_weeks": { "zh": "EpiWeek 1 ~ 53 週", "en": "EpiWeek 1 ~ 53 Weeks" },

    # ---------------- Inference Animation Steps ----------------
    "step_desc_tfm": {
        "zh": "Google TimesFM 3.0 (330M 參數) 深度基礎大模型零樣本推論中...",
        "en": "Google TimesFM 3.0 (330M Params) Foundation Model Zero-Shot Inferencing..."
    },
    "step_desc_prophet": {
        "zh": "Meta Prophet 貝氏轉折點與傅立葉季節成分擬合中...",
        "en": "Meta Prophet fitting Bayesian changepoints & Fourier seasonality..."
    },
    "step_desc_arima": {
        "zh": "Auto ARIMA (p, d, q) 網格極大似然估計與模型定階搜尋中...",
        "en": "Auto ARIMA (p, d, q) MLE grid search & AIC model selection..."
    },
    "anim_step_1": {
        "zh": "Google TimesFM 3.0 (330M 參數) 深度基礎大模型零樣本推論中...",
        "en": "Google TimesFM 3.0 (330M Params) Foundation Model Zero-Shot Inferencing..."
    },
    "anim_step_2": {
        "zh": "Meta Prophet 貝氏轉折點與傅立葉季節成分擬合中...",
        "en": "Meta Prophet fitting Bayesian changepoints & Fourier seasonality..."
    },
    "anim_step_3": {
        "zh": "Auto ARIMA (p, d, q) 網格極大似然估計與模型定階搜尋中...",
        "en": "Auto ARIMA (p, d, q) MLE grid search & AIC model selection..."
    },
    "error_insufficient_data": {
        "zh": "歷史資料長度不足（需大於 {req} 筆方可進行 {horizon} 期回測）。",
        "en": "Insufficient historical data (need > {req} records for {horizon}-step backtest)."
    },

    # ---------------- Hyperparameter Tuning ----------------
    "prophet_expander": {
        "zh": "🔧 Meta Prophet 傳染病參數微調",
        "en": "🔧 Meta Prophet Hyperparameters"
    },
    "prophet_log": {
        "zh": "Prophet 對數轉換 log(y+1)",
        "en": "Prophet Log1p Transform log(y+1)"
    },
    "prophet_log_label": {
        "zh": "Prophet 對數轉換 log(y+1)",
        "en": "Prophet Log1p Transform log(y+1)"
    },
    "prophet_log_help": {
        "zh": "強烈建議開啟：保證非負預測，貼合傳染病乘法指數傳播機制。",
        "en": "Recommended: Guarantees non-negative forecasts and captures multiplicative dynamics."
    },
    "prophet_cp_scale": {
        "zh": "轉折點先驗尺度 (prior_scale)：",
        "en": "Changepoint Prior Scale (prior_scale):"
    },
    "prophet_cp_scale_label": {
        "zh": "轉折點先驗尺度 (prior_scale)：",
        "en": "Changepoint Prior Scale (prior_scale):"
    },
    "prophet_cp_scale_help": {
        "zh": "預設 0.15。調高可更快捕捉疫情急轉直下或爆發。",
        "en": "Default: 0.15. Higher values adapt faster to sudden trend shifts."
    },
    "prophet_cp_range": {
        "zh": "轉折點歷史範圍 (changepoint_range)：",
        "en": "Changepoint History Range:"
    },
    "prophet_cp_range_label": {
        "zh": "轉折點歷史範圍 (changepoint_range)：",
        "en": "Changepoint History Range:"
    },
    "prophet_cp_range_help": {
        "zh": "預設 0.95。涵蓋至最近 95% 歷史時間，避免忽略最新波段。",
        "en": "Default: 0.95. Covers up to 95% of history to avoid lagging behind recent turns."
    },
    "arima_expander": {
        "zh": "🔧 Auto ARIMA 參數微調",
        "en": "🔧 Auto ARIMA Hyperparameters"
    },
    "arima_log": {
        "zh": "ARIMA 對數轉換 log(y+1)",
        "en": "ARIMA Log1p Transform log(y+1)"
    },
    "arima_log_label": {
        "zh": "ARIMA 對數轉換 log(y+1)",
        "en": "ARIMA Log1p Transform log(y+1)"
    },
    "arima_log_help": {
        "zh": "強烈建議開啟：保證 ARIMA 預測結果非負且防止發散。",
        "en": "Recommended: Prevents divergence and guarantees non-negative forecasts."
    },
    "arima_max_p": {
        "zh": "最大 AR 階數 (max_p)：",
        "en": "Max AR Order (max_p):"
    },
    "arima_max_p_label": {
        "zh": "最大 AR 階數 (max_p)：",
        "en": "Max AR Order (max_p):"
    },
    "arima_max_q": {
        "zh": "最大 MA 階數 (max_q)：",
        "en": "Max MA Order (max_q):"
    },
    "arima_max_q_label": {
        "zh": "最大 MA 階數 (max_q)：",
        "en": "Max MA Order (max_q):"
    },
    "arima_seasonal": {
        "zh": "啟用季節性 SARIMA",
        "en": "Enable Seasonal SARIMA"
    },
    "arima_seasonal_label": {
        "zh": "啟用季節性 SARIMA",
        "en": "Enable Seasonal SARIMA"
    },
    "arima_seasonal_help": {
        "zh": "若為日資料會以 m=7 進行週期搜尋。週資料建議關閉以加速推論。",
        "en": "Searches weekly cycle (m=7) for daily data. Disabled on weekly series for speed."
    },
    "restore_notices_btn": {
        "zh": "🔔 重新顯示系統公告說明",
        "en": "🔔 Restore System Notices"
    },
    "restore_notices_help": {
        "zh": "點擊重新顯示頂部資料處理與流行病學防護公告",
        "en": "Click to redisplay top notices on data processing and lag protection"
    },

    # ---------------- Tabs ----------------
    "tab1_title": {
        "zh": "📊 三模型走勢對比圖 (Forecast Overlay)",
        "en": "📊 3-Model Forecast Overlay"
    },
    "tab2_title": {
        "zh": "📈 評估指標計分卡 (Accuracy Scorecard)",
        "en": "📈 Evaluation Scorecard"
    },
    "tab3_title": {
        "zh": "🧠 疫情研判與三維模型洞察 (Epidemic Report)",
        "en": "🧠 Trajectory Insights & Epidemic Report"
    },
    "tab4_title": {
        "zh": "📋 預測明細表與資料匯出 (Data Export)",
        "en": "📋 Forecast Details & Data Export"
    },

    # ---------------- Tab 1: Chart & Cards ----------------
    "tab1_header": {
        "zh": "【{target}】走勢與未來 {horizon} 期三模型預測對比",
        "en": "[{target}] Trajectory & Next {horizon}-Step 3-Model Comparison"
    },
    "scale_mode_label": {
        "zh": "📐 縱軸刻度模式：",
        "en": "📐 Y-Axis Scale Mode:"
    },
    "scale_linear": {
        "zh": "線性刻度 (Linear)",
        "en": "Linear Scale"
    },
    "scale_log": {
        "zh": "對數刻度 (Log10, 放大個位數)",
        "en": "Log10 Scale (Highlight Outbreaks & Base)"
    },
    "scale_mode_help": {
        "zh": "對數刻度採 log10 映射，可讓個位數（1~10例）與千位數（1,000~2,500例）同時清晰可見，避免微小疫情被歷史大流行壓平。",
        "en": "Log10 mapping highlights single digits (1~10) and peak outbreaks (1,000~2,500) simultaneously without baseline flattening."
    },
    "trace_history": {
        "zh": "歷史觀測數據",
        "en": "Historical Observation"
    },
    "trace_tfm_ci": {
        "zh": "TimesFM 3.0 信賴區間",
        "en": "TimesFM 3.0 CI"
    },
    "trace_prophet_ci": {
        "zh": "Prophet 信賴區間",
        "en": "Prophet CI"
    },
    "trace_arima_ci": {
        "zh": "Auto ARIMA 信賴區間",
        "en": "Auto ARIMA CI"
    },
    "trace_tfm_pred": {
        "zh": "TimesFM 3.0 預測",
        "en": "TimesFM 3.0 Forecast"
    },
    "trace_prophet_pred": {
        "zh": "Prophet 預測",
        "en": "Prophet Forecast"
    },
    "trace_arima_pred": {
        "zh": "Auto ARIMA 預測",
        "en": "Auto ARIMA Forecast"
    },
    "trace_excluded": {
        "zh": "⚠️ 最新未滿期 (已排除)",
        "en": "⚠️ Latest Incomplete Period (Excluded)"
    },
    "trace_ground_truth": {
        "zh": "真實值 (Ground Truth)",
        "en": "Actual (Ground Truth)"
    },
    "forecast_origin": {
        "zh": "預測起點",
        "en": "Forecast Start"
    },
    "forecast_start_line": {
        "zh": "預測起點",
        "en": "Forecast Start"
    },
    "btn_6m": { "zh": "近半年", "en": "6M" },
    "btn_1y": { "zh": "近 1 年", "en": "1Y" },
    "btn_2y": { "zh": "近 2 年", "en": "2Y" },
    "btn_all": { "zh": "全量歷史", "en": "All" },
    "axis_ym": { "zh": "月起始日 (Month Start Date)", "en": "Month Start Date" },
    "axis_yw": { "zh": "週起始日 (Week Start Date)", "en": "Week Start Date" },
    "axis_date": { "zh": "日期 / 時間 (Date)", "en": "Date / Time" },
    "xaxis_month": { "zh": "月起始日 (Month Start Date)", "en": "Month Start Date" },
    "xaxis_week": { "zh": "週起始日 (Week Start Date)", "en": "Week Start Date" },
    "xaxis_date": { "zh": "日期 / 時間 (Date)", "en": "Date / Time" },

    # Hover templates
    "hover_hist_ym": {
        "zh": "<b>歷史觀測 (建模基準)</b><br>發病年月: %{customdata[0]}<br>月起始日: %{x}<br>數值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Historical Observation</b><br>Year-Month: %{customdata[0]}<br>Month Start: %{x}<br>Value: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_tfm_ym": {
        "zh": "<b>TimesFM 3.0 預測</b><br>預測年月: %{customdata[0]}<br>月起始日: %{x}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>TimesFM 3.0 Forecast</b><br>Projected YM: %{customdata[0]}<br>Month Start: %{x}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_pro_ym": {
        "zh": "<b>Meta Prophet 預測</b><br>預測年月: %{customdata[0]}<br>月起始日: %{x}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Meta Prophet Forecast</b><br>Projected YM: %{customdata[0]}<br>Month Start: %{x}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_ari_ym": {
        "zh": "<b>Auto ARIMA 預測</b><br>預測年月: %{customdata[0]}<br>月起始日: %{x}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Auto ARIMA Forecast</b><br>Projected YM: %{customdata[0]}<br>Month Start: %{x}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_gt_ym": {
        "zh": "<b>真實觀測值</b><br>發病年月: %{customdata[0]}<br>月起始日: %{x}<br>真實值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Ground Truth</b><br>Year-Month: %{customdata[0]}<br>Month Start: %{x}<br>Actual: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_hist_yw": {
        "zh": "<b>歷史觀測 (建模基準)</b><br>疾管署年週: %{customdata[0]}<br>週起始日: %{x}<br>數值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Historical Observation</b><br>CDC EpiWeek: %{customdata[0]}<br>Week Start: %{x}<br>Value: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_tfm_yw": {
        "zh": "<b>TimesFM 3.0 預測</b><br>預測年週: %{customdata[0]}<br>週起始日: %{x}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>TimesFM 3.0 Forecast</b><br>Projected EpiWeek: %{customdata[0]}<br>Week Start: %{x}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_pro_yw": {
        "zh": "<b>Meta Prophet 預測</b><br>預測年週: %{customdata[0]}<br>週起始日: %{x}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Meta Prophet Forecast</b><br>Projected EpiWeek: %{customdata[0]}<br>Week Start: %{x}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_ari_yw": {
        "zh": "<b>Auto ARIMA 預測</b><br>預測年週: %{customdata[0]}<br>週起始日: %{x}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Auto ARIMA Forecast</b><br>Projected EpiWeek: %{customdata[0]}<br>Week Start: %{x}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_gt_yw": {
        "zh": "<b>真實觀測值</b><br>疾管署年週: %{customdata[0]}<br>週起始日: %{x}<br>真實值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Ground Truth</b><br>CDC EpiWeek: %{customdata[0]}<br>Week Start: %{x}<br>Actual: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_hist_d": {
        "zh": "<b>歷史觀測 (建模基準)</b><br>日期: %{customdata[0]}<br>數值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Historical Observation</b><br>Date: %{customdata[0]}<br>Value: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_tfm_d": {
        "zh": "<b>TimesFM 3.0 預測</b><br>日期: %{customdata[0]}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>TimesFM 3.0 Forecast</b><br>Date: %{customdata[0]}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_pro_d": {
        "zh": "<b>Meta Prophet 預測</b><br>日期: %{customdata[0]}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Meta Prophet Forecast</b><br>Date: %{customdata[0]}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_ari_d": {
        "zh": "<b>Auto ARIMA 預測</b><br>日期: %{customdata[0]}<br>預測值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Auto ARIMA Forecast</b><br>Date: %{customdata[0]}<br>Forecast: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_gt_d": {
        "zh": "<b>真實觀測值</b><br>日期: %{customdata[0]}<br>真實值: %{customdata[1]:,.1f}<extra></extra>",
        "en": "<b>Ground Truth</b><br>Date: %{customdata[0]}<br>Actual: %{customdata[1]:,.1f}<extra></extra>"
    },
    "hover_excluded_incomplete": {
        "zh": "<b>⚠️ 初步通報數 (未滿整期 / 已排除建模)</b><br>{period_label}: {period_val}<br>數值: {val:,.1f}<br><i>(因統計未滿期，系統已排除以防止模型誤判)</i><extra></extra>",
        "en": "<b>⚠️ Preliminary Report (Partial / Excluded)</b><br>{period_label}: {period_val}<br>Value: {val:,.1f}<br><i>(Excluded to prevent model distortion due to reporting lag)</i><extra></extra>"
    },

    # Summary Cards
    "card_tfm_title": { "zh": "Google TimesFM 3.0", "en": "Google TimesFM 3.0" },
    "card_prophet_title": { "zh": "Meta Prophet (優化版)", "en": "Meta Prophet (Optimized)" },
    "card_arima_title": { "zh": "Auto ARIMA", "en": "Auto ARIMA" },
    "card_mean": { "zh": "{horizon} 期均值", "en": "{horizon}-Step Mean" },
    "card_range": { "zh": "預測範圍", "en": "Range" },
    "card_latency": { "zh": "推論耗時", "en": "Inference Time" },
    "card_order_aic": { "zh": "階數/AIC", "en": "Order/AIC" },

    # Seasonality Overlay
    "overlay_expander_title": {
        "zh": "📊 歷年同{unit}季節性流行趨勢疊加分析 (跨年度同{unit}對比，解決大流行年壓平問題)",
        "en": "📊 Multi-Year Same-{unit} Seasonality Overlay (Resolves Outbreak Compression)"
    },
    "overlay_guidance": {
        "zh": "💡 **流行病學分析指引：** 將各年份固定以 **{range}** 進行重疊對比。可點擊下方圖例中的年份以**暫時隱藏該年份**，圖表 Y 軸將會自動重縮放，即刻以最大解析度檢視近年之波動特徵！",
        "en": "💡 **Epidemiological Guideline:** Overlays all years across a standardized **{range}** timeline. Click on any year in the legend below to **temporarily toggle off outlier years**, instantly rescaling the Y-axis to inspect subtle recent patterns!"
    },
    "overlay_log_chk": {
        "zh": "疊加圖啟用對數尺度 (Log10)",
        "en": "Enable Overlay Log Scale (Log10)"
    },
    "overlay_chart_title": {
        "zh": "歷年同{unit}確診趨勢對比 ({range})",
        "en": "Historical Same-{unit} Trend Comparison ({range})"
    },
    "overlay_trace_name": {
        "zh": "{yr} 年 (峰值 {peak} {cases} / 年計 {sum} {cases})",
        "en": "{yr} (Peak {peak} {cases} / Annual {sum} {cases})"
    },
    "overlay_hover": {
        "zh": "<b>{yr} 年</b><br>{unit}份: 第 %{{x}} {unit}<br>確定病例數: %{{customdata:,.0f}} {cases}<extra></extra>",
        "en": "<b>{yr}</b><br>{unit}: #{x}<br>Cases: %{customdata:,.0f} {cases}<extra></extra>"
    },
    "th_year": { "zh": "年份", "en": "Year" },
    "th_stat_periods": { "zh": "統計{unit}數", "en": "Periods Count ({unit})" },
    "th_annual_sum": { "zh": "年度累計個案數", "en": "Annual Cumulative Sum" },
    "th_peak_val": { "zh": "單{unit}最高峰值", "en": "Single-{unit} Peak Value" },
    "th_peak_timing": { "zh": "發生峰值{unit}次", "en": "Peak Period Occurrence" },
    "th_mean_val": { "zh": "{unit}平均病例數", "en": "Mean Cases per {unit}" },
    "prefix_period": { "zh": "第 ", "en": "#" },
    "scale_linear_suffix": { "zh": "(線性)", "en": "(Linear)" },
    "scale_log_suffix": { "zh": "(對數 Log10)", "en": "(Log10)" },

    # ---------------- Tab 2: Metrics & Stats ----------------
    "tab2_header": {
        "zh": "三模型指標對比計分卡 (Accuracy Scorecard)",
        "en": "3-Model Accuracy Scorecard"
    },
    "backtest_success_msg": {
        "zh": "當前處於「歷史回測模式」，已使用歷史保留之最後 {horizon} 期真實值進行嚴謹的精準度評估！",
        "en": "Currently in 'Historical Backtest Mode'. Models evaluated against held-out ground truth over the last {horizon} periods!"
    },
    "future_mode_info": {
        "zh": "💡 目前處於「未來預測模式」（外推未知未來）。若需查看各項誤差量化指標（MAE, RMSE, MAPE, 峰值偏差），請在左側側邊欄切換為 **「🧪 歷史回測模式」**。",
        "en": "💡 Currently in 'Future Forecast Mode' (extrapolating into unknown future). To inspect quantitative error metrics (MAE, RMSE, MAPE, Peak Error), switch to **'🧪 Historical Backtest Mode'** in the sidebar."
    },
    "metric_mae": { "zh": "MAE (平均絕對誤差)", "en": "MAE (Mean Absolute Error)" },
    "metric_rmse": { "zh": "RMSE (均方根誤差)", "en": "RMSE (Root Mean Squared Error)" },
    "metric_mape": { "zh": "MAPE (平均絕對百分比誤差)", "en": "MAPE (Mean Absolute Percentage Error)" },
    "metric_smape": { "zh": "SMAPE (對稱百分比誤差)", "en": "SMAPE (Symmetric MAPE)" },
    "metric_wape": { "zh": "WAPE (加權絕對百分比誤差)", "en": "WAPE (Weighted APE)" },
    "metric_dir_acc": { "zh": "趨勢方向準確率 (Directional Acc)", "en": "Directional Accuracy" },
    "metric_peak_err": { "zh": "峰值預測偏差率 (Peak Error)", "en": "Peak Value Error" },
    "metric_timing_err": { "zh": "達峰時間偏差 (Timing Error)", "en": "Peak Timing Error" },
    "metric_ci_cov": { "zh": "信賴區間覆蓋率 (CI Coverage)", "en": "CI Coverage Probability" },
    "pref_lower": { "zh": "越低越好", "en": "Lower is better" },
    "pref_higher": { "zh": "越高越好", "en": "Higher is better" },
    "th_eval_metric": { "zh": "評估指標", "en": "Evaluation Metric" },
    "th_winner": { "zh": "優勝模型", "en": "Best Model" },
    "th_criteria": { "zh": "指標準則", "en": "Criteria" },
    "bar_chart_header": { "zh": "模型指標可視化柱狀圖", "en": "Model Metric Comparative Bar Chart" },
    "bar_toggle_label": { "zh": "切換指標對比：", "en": "Select Metric to Compare:" },
    "bar_mae": { "zh": "MAE (平均絕對誤差 - 越低越好)", "en": "MAE (Mean Absolute Error - Lower is Better)" },
    "bar_mape": { "zh": "MAPE (%) (百分比誤差 - 越低越好)", "en": "MAPE (%) (Percentage Error - Lower is Better)" },
    "bar_dir_acc": { "zh": "趨勢方向準確率 (%) (越高越好)", "en": "Directional Accuracy (%) (Higher is Better)" },
    "bar_title_mae": { "zh": "MAE 平均絕對誤差對比 (次/例，越低越好)", "en": "MAE Mean Absolute Error Comparison (Lower is Better)" },
    "bar_title_mape": { "zh": "MAPE 平均絕對百分比誤差 (%) 對比 (越低越好)", "en": "MAPE Mean Absolute Percentage Error (%) (Lower is Better)" },
    "bar_title_dir_acc": { "zh": "趨勢方向準確率 (%) 對比 (越高越好)", "en": "Directional Accuracy (%) Comparison (Higher is Better)" },
    "th_stat_feature": { "zh": "統計特徵", "en": "Statistical Feature" },
    "stat_horizon": { "zh": "預測期數", "en": "Forecast Steps" },
    "stat_peak": { "zh": "預測最大值 (峰值)", "en": "Projected Peak (Max)" },
    "stat_min": { "zh": "預測最小值", "en": "Projected Minimum" },
    "stat_mean": { "zh": "預測平均值", "en": "Projected Mean" },
    "stat_sum": { "zh": "預測總和", "en": "Projected Sum" },
    "stat_latency": { "zh": "推論延遲 (秒)", "en": "Inference Latency (s)" },

    # ---------------- Tab 3: Insights Report ----------------
    "tab3_header": {
        "zh": "趨勢研判與三維模型深度洞察報告",
        "en": "Trajectory Insights & 3-Model Epidemic Report"
    },
    "stage_strong_up": {
        "zh": "📈 **強勁上升階段** (走勢向上顯著突破)",
        "en": "📈 **Strong Uptrend** (Significant upward momentum)"
    },
    "stage_mild_up": {
        "zh": "↗️ **溫和上升階段** (呈漸進增長態勢)",
        "en": "↗️ **Moderate Uptrend** (Gradual progressive increase)"
    },
    "stage_plateau": {
        "zh": "⏸️ **高原震盪 / 持平盤整階段**",
        "en": "⏸️ **Consolidation / Plateau** (Sideways stability)"
    },
    "stage_mild_down": {
        "zh": "↘️ **穩定趨緩 / 漸次回檔階段**",
        "en": "↘️ **Steady Pullback / Softening Trend**"
    },
    "stage_rapid_down": {
        "zh": "📉 **快速下降 / 收斂階段**",
        "en": "📉 **Rapid Decline / Downward Convergence**"
    },
    "prefix_ym": { "zh": "年月", "en": "YM" },
    "prefix_yw": { "zh": "年週", "en": "EpiWeek" },
    "report_sec1_title": {
        "zh": "未來走向研判 (Forecast Trajectory)",
        "en": "Forecast Trajectory & Trend Assessment"
    },
    "report_baseline_title": {
        "zh": "歷史完整建模基準點",
        "en": "Historical Baseline Point"
    },
    "report_assessment": {
        "zh": "研判",
        "en": "Assessment"
    },
    "report_pred_step": {
        "zh": "未來第 {horizon} 期預計為",
        "en": "Step +{horizon} projected at"
    },
    "report_change_rate": {
        "zh": "變動率",
        "en": "Change Rate"
    },
    "report_sec2_title": {
        "zh": "時序預測關鍵機制解析",
        "en": "Key Methodological & Protective Mechanisms"
    },
    "report_p1_title": {
        "zh": "最新一期不完整數據排除（Right-Censored Lag Protection）",
        "en": "Right-Censored Reporting Lag Protection"
    },
    "report_phenom": {
        "zh": "現象",
        "en": "Observation"
    },
    "report_protect": {
        "zh": "防護效應",
        "en": "Safeguard"
    },
    "report_p1_phenom": {
        "zh": "流行病監測最新一期常因資料擷取時尚未滿期，或基層醫療機構通報遞延，呈現斷崖式偏低。",
        "en": "In surveillance time series, the most recent period is often incomplete due to reporting lag, manifesting as an artificial drop."
    },
    "report_p1_protect": {
        "zh": "若不排除此點，模型將誤判為「疫情崩跌」而直線下探 0 例；排除後以最近完整期別為基準，模型能預測出自然平滑演變曲線！若為已結算之日報或金融走勢數據，可直接納入最新記錄。",
        "en": "Excluding partial periods prevents models from diagnosing an artificial crash to zero, producing realistic trajectories. For finalized financial or daily series, this can be seamlessly disabled."
    },
    "report_p2_title": {
        "zh": "自適應時間頻率與疾管署 EpiWeek 引擎（Adaptive Temporal Engine）",
        "en": "Adaptive Temporal Engine & Taiwan CDC EpiWeek Standards"
    },
    "report_p2_desc": {
        "zh": "內建台灣疾管署官方週期數學演算法，自動識別「發病年週 (EpiWeek)」、「發病年月 (Year-Month)」、日報與一般日期序列，無縫適配多層次時間顆粒度。",
        "en": "Built-in exact mathematical rules accurately resolve Taiwan CDC / MMWR EpiWeeks, Year-Months, and daily series without static calendar expiration."
    },
    "report_p3_title": {
        "zh": "即時 OpenData 串接與備援體系（Live OpenData & Offline Resilience）",
        "en": "Live OpenData Synchronization & Offline Resilience"
    },
    "report_p3_desc": {
        "zh": "支援央行匯率及外部自訂 CSV 固定網址即時同步，並內建 SSRF 安全防護與離線備援機制，確保系統在內外網各類環境均能穩健運作。",
        "en": "Supports live feeds for Central Bank exchange rates and custom CSV URLs with SSRF protection and instant fallback to local datasets."
    },

    # ---------------- Tab 4: Export Table ----------------
    "tab4_header": {
        "zh": "未來 {horizon} 期預測明細數據表",
        "en": "Detailed Forecast Data Table (Next {horizon} Steps)"
    },
    "col_step": { "zh": "期數 (Step)", "en": "Step" },
    "col_pred_month_start": { "zh": "預測日期 (月起始日)", "en": "Forecast Date (Month Start)" },
    "col_pred_week_start": { "zh": "預測日期 (週起始日)", "en": "Forecast Date (Week Start)" },
    "col_pred_date": { "zh": "預測日期 (Date)", "en": "Forecast Date (Date)" },
    "col_cdc_ym": { "zh": "疾管署年月 (Year-Month)", "en": "CDC Year-Month" },
    "col_cdc_yw": { "zh": "疾管署年週 (Year-Week)", "en": "CDC Year-Week (EpiWeek)" },
    "col_tfm_pred": { "zh": "TimesFM 3.0 預測值", "en": "TimesFM 3.0 Forecast" },
    "col_tfm_ci_low": { "zh": "TimesFM 信賴下限", "en": "TimesFM Lower CI" },
    "col_tfm_ci_high": { "zh": "TimesFM 信賴上限", "en": "TimesFM Upper CI" },
    "col_pro_pred": { "zh": "Prophet 預測值", "en": "Prophet Forecast" },
    "col_pro_ci_low": { "zh": "Prophet 信賴下限", "en": "Prophet Lower CI" },
    "col_pro_ci_high": { "zh": "Prophet 信賴上限", "en": "Prophet Upper CI" },
    "col_ari_pred": { "zh": "Auto ARIMA 預測值", "en": "Auto ARIMA Forecast" },
    "col_ari_ci_low": { "zh": "ARIMA 信賴下限", "en": "ARIMA Lower CI" },
    "col_ari_ci_high": { "zh": "ARIMA 信賴上限", "en": "ARIMA Upper CI" },
    "col_actual": { "zh": "真實值 (Actual)", "en": "Actual (Ground Truth)" },
    "col_tfm_err": { "zh": "TimesFM 絕對誤差", "en": "TimesFM Absolute Error" },
    "col_pro_err": { "zh": "Prophet 絕對誤差", "en": "Prophet Absolute Error" },
    "col_ari_err": { "zh": "ARIMA 絕對誤差", "en": "ARIMA Absolute Error" },
    "btn_download_csv": {
        "zh": "📥 下載 {horizon} 期預測明細報表 (CSV - 安全防注入格式)",
        "en": "📥 Download {horizon}-Step Forecast Report (CSV - Safe Format)"
    },
    "btn_download_help": {
        "zh": "下載包含三大模型預測值、疾管署年週與信賴區間的完整表格（已通過 OWASP CSV Formula Injection 安全防護過濾）",
        "en": "Download full forecast table with 3-model predictions, CDC EpiWeeks, and confidence intervals (OWASP CSV Injection Protected)"
    },
}


def t(key: str, lang: str = "zh", **kwargs) -> str:
    """
    Translates a given key into the target language ('zh' or 'en').
    Supports keyword interpolation using kwargs.
    Falls back to 'zh' if key is missing in 'en', or returns key if not found.
    """
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key

    text = entry.get(lang)
    if text is None:
        text = entry.get("zh", key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text

    return text


def get_dataset_choices(lang: str = "zh") -> Dict[str, str]:
    """Returns mapping from internal dataset key to localized display name."""
    return {
        "enterovirus": t("ds_enterovirus", lang),
        "dengue": t("ds_dengue", lang),
        "scrub_typhus": t("ds_scrub_typhus", lang),
        "covid": t("ds_covid", lang),
        "flu": t("ds_flu", lang),
        "cbc_forex": t("ds_cbc_forex", lang),
        "custom_url": t("ds_custom_url", lang),
        "upload": t("ds_upload", lang),
        "paste": t("ds_paste", lang),
    }


def get_freq_choices(lang: str = "zh") -> Dict[str, str]:
    """Returns mapping from pandas frequency code to localized display name."""
    return {
        "W": t("freq_weekly", lang),
        "D": t("freq_daily", lang),
        "M": t("freq_monthly", lang),
    }


def get_eval_mode_choices(lang: str = "zh") -> Dict[str, str]:
    """Returns mapping from internal eval mode code to localized display name."""
    return {
        "future": t("eval_mode_future", lang),
        "backtest": t("eval_mode_backtest", lang),
    }


def get_time_window_choices(min_year: int, max_year: int, lang: str = "zh") -> List[Tuple[Any, str]]:
    """
    Generates localized list of (window_id, label) tuples.
    window_id is 'all' or year integer.
    """
    options: List[Tuple[Any, str]] = [("all", t("time_window_all", lang))]
    for y in range(min_year + 1, max_year + 1):
        options.append((y, t("time_window_from", lang, year=y)))
    return options


COLUMN_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Dates / Time periods
    "就診年週": { "zh": "就診年週", "en": "Consultation EpiWeek" },
    "發病年週": { "zh": "發病年週", "en": "Onset EpiWeek" },
    "發病年月": { "zh": "發病年月", "en": "Onset Year-Month" },
    "發病年月日": { "zh": "發病年月日", "en": "Onset Date" },
    "日期": { "zh": "日期", "en": "Date" },
    "時間": { "zh": "時間", "en": "Time" },
    "年週": { "zh": "年週", "en": "Year-Week" },
    "週次": { "zh": "週次", "en": "Week Number" },
    "週別": { "zh": "週別", "en": "Week Number" },
    "年月": { "zh": "年月", "en": "Year-Month" },
    "年月日": { "zh": "年月日", "en": "Date" },
    "年份": { "zh": "年份", "en": "Year" },
    "月份": { "zh": "月份", "en": "Month" },
    "交易日期": { "zh": "交易日期", "en": "Trading Date" },
    "資料日期": { "zh": "資料日期", "en": "Data Date" },

    # Target values / Metrics
    "全國": { "zh": "全國", "en": "National Total" },
    "確定病例數": { "zh": "確定病例數", "en": "Confirmed Cases" },
    "收盤匯率": { "zh": "收盤匯率", "en": "Closing Exchange Rate" },
    "匯率": { "zh": "匯率", "en": "Exchange Rate" },
    "個案數": { "zh": "個案數", "en": "Cases Count" },
    "確診數": { "zh": "確診數", "en": "Confirmed Cases" },
    "病例數": { "zh": "病例數", "en": "Cases Count" },
    "就診人次": { "zh": "就診人次", "en": "Outpatient Visits" },
    "門急診就診人次": { "zh": "門急診就診人次", "en": "Outpatient & ER Visits" },
    "急診就診人次": { "zh": "急診就診人次", "en": "ER Visits" },
    "人數": { "zh": "人數", "en": "Persons / Visits" },
    "人次": { "zh": "人次", "en": "Visits Count" },
    "數值": { "zh": "數值", "en": "Value" },
    "收盤價": { "zh": "收盤價", "en": "Closing Price" },
}


def get_column_display_name(col_name: Any, lang: str = "zh") -> str:
    """Translates known column names to the target language, or returns original name."""
    if not col_name:
        return str(col_name) if col_name is not None else ""
    s = str(col_name).strip()
    entry = COLUMN_TRANSLATIONS.get(s)
    if entry:
        return entry.get(lang, s)
    return s
