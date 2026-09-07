"""
Interactive transition and loading animations for epidemic model inference.
Provides multiple distinct animated themes to keep users entertained during waiting times.
"""

import random
from typing import Optional, Dict, Any


ANIMATION_THEMES_ZH = {
    "random": "🎲 每次隨機驚喜 (Random Surprise)",
    "race": "🏃‍♂️ 傳染病大賽跑 (Model Grand Prix)",
    "radar": "🧬 流行病學量子雷達 (Bio-Radar Sweep)",
    "chrono_warp": "🌌 超時空賽博神經網 (Chrono Warp Grid)",
    "coffee": "☕ 防疫研究員咖啡時光 (Coffee Break)"
}

ANIMATION_THEMES_EN = {
    "random": "🎲 Random Surprise",
    "race": "🏃‍♂️ Model Grand Prix",
    "radar": "🧬 Bio-Radar Sweep",
    "chrono_warp": "🌌 Chrono Warp Grid",
    "coffee": "☕ Coffee Break"
}

ANIMATION_THEMES = ANIMATION_THEMES_ZH


def get_animation_themes(lang: str = "zh") -> Dict[str, str]:
    """Returns localized animation theme dictionary."""
    return ANIMATION_THEMES_EN if lang == "en" else ANIMATION_THEMES_ZH


QUOTES_ZH = [
    ("🤖", "TimesFM 3.0 正在從 3.3 億個基礎神經元參數中解碼時間序列特徵..."),
    ("🧙‍♂️", "Prophet 巫師正在召喚傅立葉級數與貝氏轉折點，感應流行波長..."),
    ("📐", "Auto ARIMA 正在平行宇宙中極速遍歷搜尋最佳 (p, d, q) 階數與 AIC..."),
    ("🦟", "登革熱病媒蚊表示：在大數據時代，連我要叮幾個人都被 AI 算得明明白白！"),
    ("👶", "腸病毒正在籌劃開學季大派對，AI 模型正全力推演雙峰防禦曲線！"),
    ("🦗", "恙蟲在草叢裡竊竊私語：這 27 年來的發病月走勢居然被完全看穿了！"),
    ("🧼", "防疫小冷知識：衛福部建議內外夾弓大立腕洗手 20 秒，而本模型推論只要 5 秒！"),
    ("☕", "請稍候一口咖啡的時間，三核心 AI 正在推演未來時空的流行病學軌跡..."),
    ("🛡️", "已自動啟動未滿期資料防護罩，杜絕通報遞延導致的斷崖式誤判！"),
    ("📊", "白噪音正在被過濾，殘差序列正在被馴服，信賴區間防護網正在全力展開..."),
    ("🧬", "MMWR 流行病學年週數學引擎運轉中，跨入 2050 年依然秒級精準推算！"),
    ("🚀", "CPU 與神經網絡引擎正在全速進行矩陣乘法運算，精彩預測即刻出爐！")
]

QUOTES_EN = [
    ("🤖", "TimesFM 3.0 is decoding temporal patterns from 330M pre-trained foundation parameters..."),
    ("🧙‍♂️", "Prophet is channeling Fourier harmonics and Bayesian changepoints to forecast trajectory..."),
    ("📐", "Auto ARIMA is traversing parallel universes to optimize (p, d, q) orders and AIC..."),
    ("🦟", "Vector surveillance alert: in the era of Big Data, transmission trends are forecasted by AI!"),
    ("👶", "Epidemiological model adapting to multi-wave seasonal spikes and school-year cycles..."),
    ("🦗", "Analyzing multi-decade longitudinal surveillance records to uncover climate seasonality..."),
    ("☕", "Take a sip of coffee while the 3-engine ensemble projects the future trajectory..."),
    ("🛡️", "Reporting lag protection active: shielding models against incomplete period distortion!"),
    ("📊", "Filtering white noise, taming residual variance, unfolding confidence intervals..."),
    ("🚀", "CPU & neural tensor engines computing matrix multiplications at full throttle!"),
]

QUOTES = QUOTES_ZH


def get_random_quote(lang: str = "zh") -> tuple:
    pool = QUOTES_EN if lang == "en" else QUOTES_ZH
    return random.choice(pool)


def render_loading_card(
    theme_key: str = "random",
    step_num: int = 1,
    step_desc: Optional[str] = None,
    quote: Optional[tuple] = None,
    lang: str = "zh"
) -> str:
    """
    Renders a complete, responsive, self-contained HTML/CSS animated card.
    """
    if theme_key == "random" or theme_key not in ["race", "radar", "chrono_warp", "coffee"]:
        available = ["race", "radar", "chrono_warp", "coffee"]
        theme_key = random.choice(available)

    if step_desc is None:
        step_desc = "Google TimesFM 3.0 Foundation Model Inferencing..." if lang == "en" else "Google TimesFM 3.0 基礎大模型推論中..."

    if quote is None:
        quote = get_random_quote(lang=lang)

    q_icon, q_text = quote

    # Generate theme-specific inner HTML
    if theme_key == "race":
        anim_content = _get_race_html(lang=lang)
        badge_text = "🏃‍♂️ Model Grand Prix: 3-Way Forecasting Race" if lang == "en" else "🏃‍♂️ 傳染病大賽跑：三模型巔峰競速"
        theme_color = "#1E88E5"
    elif theme_key == "radar":
        anim_content = _get_radar_html(lang=lang)
        badge_text = "🧬 Quantum Bio-Radar: Dynamic Pathogen Scan" if lang == "en" else "🧬 流行病學量子雷達：病原體動態掃描"
        theme_color = "#00ACC1"
    elif theme_key == "chrono_warp":
        anim_content = _get_warp_html(lang=lang)
        badge_text = "🌌 Chrono Warp Grid: 330M Parameter Trajectory" if lang == "en" else "🌌 超時空賽博神經網：330M 參數時序穿梭"
        theme_color = "#7C4DFF"
    else: # coffee
        anim_content = _get_coffee_html(lang=lang)
        badge_text = "☕ Coffee Break: Brewing AI Forecasts" if lang == "en" else "☕ 防疫研究員咖啡時光：智慧萃取中"
        theme_color = "#FB8C00"

    # Step indicators
    steps = [
        ("🔵 1. TimesFM 3.0", step_num >= 1, step_num == 1),
        ("🟣 2. Meta Prophet", step_num >= 2, step_num == 2),
        ("🟢 3. Auto ARIMA", step_num >= 3, step_num == 3),
    ]
    step_pills = ""
    for label, is_done_or_active, is_active in steps:
        if is_active:
            pill_style = "background: linear-gradient(135deg, #1E88E5, #43A047); color: #fff; font-weight: 700; box-shadow: 0 0 10px rgba(30,136,229,0.5); transform: scale(1.05);"
            dot = "⚡"
        elif is_done_or_active:
            pill_style = "background: #E8F5E9; color: #2E7D32; font-weight: 600; border: 1px solid #A5D6A7;"
            dot = "✓"
        else:
            pill_style = "background: #F5F5F5; color: #9E9E9E; border: 1px solid #E0E0E0;"
            dot = "○"
        step_pills += f'<div style="padding: 6px 14px; border-radius: 20px; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; transition: all 0.3s; {pill_style}"><span>{dot}</span><span>{label}</span></div>'

    stage_label = "Current Stage: " if lang == "en" else "當前階段："
    quote_label = "Live AI & Epidemiological Insights: " if lang == "en" else "即時防疫與 AI 小小語錄："

    full_html = f"""
    <style>
    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 15px rgba(30, 136, 229, 0.2), inset 0 0 15px rgba(30, 136, 229, 0.05); }}
        50% {{ box-shadow: 0 0 28px rgba(124, 77, 255, 0.4), inset 0 0 20px rgba(124, 77, 255, 0.1); }}
        100% {{ box-shadow: 0 0 15px rgba(30, 136, 229, 0.2), inset 0 0 15px rgba(30, 136, 229, 0.05); }}
    }}
    @keyframes progressWiggle {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes spinWheel {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    .anim-card-container {{
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border: 1.5px solid #dbeafe;
        border-radius: 18px;
        padding: 24px;
        margin: 18px 0 28px 0;
        animation: pulseGlow 4s infinite ease-in-out;
        position: relative;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .anim-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 18px;
    }}
    .anim-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 136, 229, 0.08);
        border: 1px solid rgba(30, 136, 229, 0.25);
        color: {theme_color};
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }}
    .anim-step-desc {{
        font-size: 15px;
        font-weight: 600;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .anim-progress-bar {{
        width: 100%;
        height: 6px;
        background: #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
        margin: 14px 0;
    }}
    .anim-progress-fill {{
        width: {step_num * 33.33}%;
        height: 100%;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF, #43A047, #FB8C00);
        background-size: 300% 300%;
        animation: progressWiggle 2s infinite ease-in-out;
        border-radius: 8px;
        transition: width 0.4s ease;
    }}
    .anim-quote-box {{
        margin-top: 16px;
        padding: 12px 16px;
        background: rgba(241, 245, 249, 0.7);
        border-left: 4px solid {theme_color};
        border-radius: 8px;
        font-size: 13.5px;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 10px;
        line-height: 1.5;
    }}
    </style>

    <div class="anim-card-container" style="border-left-color: {theme_color};">
        <div class="anim-card-header">
            <span class="anim-badge">{badge_text}</span>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                {step_pills}
            </div>
        </div>

        <div class="anim-step-desc">
            <span style="display: inline-block; animation: spinWheel 1.5s infinite linear;">⚙️</span>
            <span><b>{stage_label}</b>{step_desc}</span>
        </div>

        <div class="anim-progress-bar">
            <div class="anim-progress-fill"></div>
        </div>

        <!-- Theme Animation Canvas -->
        <div style="width: 100%; min-height: 135px; display: flex; align-items: center; justify-content: center; background: #0b1120; border-radius: 14px; padding: 10px 14px; margin: 10px 0; position: relative; overflow: hidden; box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);">
            {anim_content}
        </div>

        <div class="anim-quote-box">
            <span style="font-size: 20px; flex-shrink: 0;">{q_icon}</span>
            <span><b>{quote_label}</b>{q_text}</span>
        </div>
    </div>
    """
    cleaned_lines = [line.strip() for line in full_html.split('\n') if line.strip()]
    return "\n".join(cleaned_lines)


# ---------------- THEME 1: RACE (傳染病大賽跑) ----------------

def _get_race_html(lang: str = "zh") -> str:
    html = """
    <style>
    @keyframes racerBob {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-3px) scale(1.02); }
    }
    @keyframes flameFlicker {
        0%, 100% { opacity: 0.7; transform: scaleX(0.8); }
        50% { opacity: 1; transform: scaleX(1.3); }
    }
    @keyframes leadShift1 {
        0%, 100% { left: 58%; }
        50% { left: 68%; }
    }
    @keyframes leadShift2 {
        0%, 100% { left: 40%; }
        50% { left: 52%; }
    }
    @keyframes leadShift3 {
        0%, 100% { left: 25%; }
        50% { left: 38%; }
    }
    @keyframes leadShift4 {
        0%, 100% { left: 74%; }
        50% { left: 82%; }
    }
    .race-track {
        width: 100%;
        height: 125px;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        padding: 4px 0;
    }
    .race-lane {
        width: 100%;
        height: 26px;
        border-bottom: 1px dashed rgba(255,255,255,0.18);
        position: relative;
        display: flex;
        align-items: center;
    }
    .race-racer {
        position: absolute;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        animation: racerBob 0.6s infinite ease-in-out;
        filter: drop-shadow(0 2px 5px rgba(0,0,0,0.6));
    }
    .race-flame {
        display: inline-block;
        animation: flameFlicker 0.25s infinite;
        transform-origin: right center;
    }
    .finish-banner {
        position: absolute;
        right: 12px;
        top: 0;
        bottom: 0;
        width: 4px;
        background: repeating-linear-gradient(0deg, #ffffff, #ffffff 6px, #1e293b 6px, #1e293b 12px);
        box-shadow: 0 0 10px #ffffff;
        z-index: 10;
    }
    .finish-tag {
        position: absolute;
        right: 18px;
        top: 4px;
        font-size: 10px;
        color: #facc15;
        font-weight: 800;
        background: rgba(0,0,0,0.6);
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #facc15;
        z-index: 11;
    }
    </style>

    <div class="race-track">
        <div class="finish-banner"></div>
        <div class="finish-tag">🎯 預測終點線</div>

        <!-- Lane 1: TimesFM 3.0 -->
        <div class="race-lane">
            <div class="race-racer" style="animation: leadShift1 3s infinite ease-in-out, racerBob 0.4s infinite;">
                <span style="font-size: 17px;">🤖</span>
                <span class="race-flame" style="font-size: 13px;">🔥</span>
                <span style="font-size: 10.5px; font-weight: 700; color: #60a5fa; background: rgba(30,136,229,0.3); padding: 1px 6px; border-radius: 10px;">TimesFM 3.0</span>
            </div>
        </div>

        <!-- Lane 2: Meta Prophet -->
        <div class="race-lane">
            <div class="race-racer" style="animation: leadShift2 3.5s infinite ease-in-out, racerBob 0.5s infinite;">
                <span style="font-size: 17px;">🧙‍♂️</span>
                <span class="race-flame" style="font-size: 13px;">✨</span>
                <span style="font-size: 10.5px; font-weight: 700; color: #c084fc; background: rgba(142,36,170,0.3); padding: 1px 6px; border-radius: 10px;">Meta Prophet</span>
            </div>
        </div>

        <!-- Lane 3: Auto ARIMA -->
        <div class="race-lane">
            <div class="race-racer" style="animation: leadShift3 4s infinite ease-in-out, racerBob 0.45s infinite;">
                <span style="font-size: 17px;">📐</span>
                <span class="race-flame" style="font-size: 13px;">⚡</span>
                <span style="font-size: 10.5px; font-weight: 700; color: #4ade80; background: rgba(67,160,71,0.3); padding: 1px 6px; border-radius: 10px;">Auto ARIMA</span>
            </div>
        </div>

        <!-- Lane 4: Pathogen Escaping -->
        <div class="race-lane">
            <div class="race-racer" style="animation: leadShift4 2.8s infinite ease-in-out, racerBob 0.35s infinite;">
                <span style="font-size: 17px;">🦠</span>
                <span style="font-size: 12px;">💦</span>
                <span style="font-size: 10.5px; font-weight: 700; color: #f87171; background: rgba(229,57,53,0.3); padding: 1px 6px; border-radius: 10px;">傳染病高峰</span>
            </div>
        </div>
    </div>
    """
    if lang == "en":
        html = html.replace("🎯 預測終點線", "🎯 Forecast Finish Line").replace("傳染病高峰", "Epidemic Outbreak")
    return html


# ---------------- THEME 2: RADAR (流行病學量子雷達) ----------------

def _get_radar_html(lang: str = "zh") -> str:
    html = """
    <style>
    @keyframes radarRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes blipGlow {
        0%, 100% { transform: scale(1); opacity: 0.2; }
        50% { transform: scale(1.8); opacity: 1; box-shadow: 0 0 12px #22d3ee; }
    }
    @keyframes blipAlert {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(2.2); opacity: 1; box-shadow: 0 0 15px #f43f5e; }
    }
    .radar-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 28px;
        width: 100%;
        height: 125px;
    }
    .radar-dish {
        width: 105px;
        height: 105px;
        border-radius: 50%;
        border: 2px solid #0891b2;
        position: relative;
        background: radial-gradient(circle, #083344 0%, #020617 75%);
        box-shadow: 0 0 18px rgba(6, 182, 212, 0.35);
        overflow: hidden;
        flex-shrink: 0;
    }
    .radar-ring {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 50%;
    }
    .radar-crosshair-v {
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        width: 1px;
        background: rgba(6, 182, 212, 0.3);
    }
    .radar-crosshair-h {
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 1px;
        background: rgba(6, 182, 212, 0.3);
    }
    .radar-sweep-needle {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: conic-gradient(from 0deg at 50% 50%, rgba(6, 182, 212, 0.65) 0deg, rgba(6, 182, 212, 0) 65deg);
        border-radius: 50%;
        animation: radarRotate 2.5s infinite linear;
        pointer-events: none;
    }
    .radar-blip {
        position: absolute;
        width: 5px;
        height: 5px;
        border-radius: 50%;
    }
    .radar-info-panel {
        display: flex;
        flex-direction: column;
        gap: 6px;
        color: #e2e8f0;
        font-size: 11.5px;
    }
    .radar-stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(6, 182, 212, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
    }
    </style>

    <div class="radar-wrapper">
        <div class="radar-dish">
            <div class="radar-ring" style="width: 28px; height: 28px;"></div>
            <div class="radar-ring" style="width: 58px; height: 58px;"></div>
            <div class="radar-ring" style="width: 88px; height: 88px;"></div>
            <div class="radar-crosshair-v"></div>
            <div class="radar-crosshair-h"></div>
            <div class="radar-sweep-needle"></div>

            <!-- Target Blips -->
            <div class="radar-blip" style="top: 22px; left: 32px; background: #22d3ee; animation: blipGlow 2.5s infinite;"></div>
            <div class="radar-blip" style="top: 68px; left: 72px; background: #f43f5e; animation: blipAlert 1.8s infinite;"></div>
            <div class="radar-blip" style="top: 36px; left: 76px; background: #22d3ee; animation: blipGlow 3s infinite;"></div>
            <div class="radar-blip" style="top: 76px; left: 26px; background: #4ade80; animation: blipGlow 2.1s infinite;"></div>
        </div>

        <div class="radar-info-panel">
            <div class="radar-stat-chip" style="color: #22d3ee;">
                <span>📡 雷達波束：</span><span>360° 流行病學監視中</span>
            </div>
            <div class="radar-stat-chip" style="color: #f43f5e;">
                <span>⚠️ 異常偵測：</span><span>病原體波形自動鎖定</span>
            </div>
            <div class="radar-stat-chip" style="color: #4ade80;">
                <span>🎯 空間網絡：</span><span>EpiWeek / 月度精準校驗</span>
            </div>
        </div>
    </div>
    """
    if lang == "en":
        html = (
            html.replace("<span>📡 雷達波束：</span><span>360° 流行病學監視中</span>", "<span>📡 Radar Beam: </span><span>360° Surveillance Active</span>")
            .replace("<span>⚠️ 異常偵測：</span><span>病原體波形自動鎖定</span>", "<span>⚠️ Pathogen Alert: </span><span>Waveform Auto-Locked</span>")
            .replace("<span>🎯 空間網絡：</span><span>EpiWeek / 月度精準校驗</span>", "<span>🎯 Spatiotemporal Grid: </span><span>EpiWeek / Monthly Calibrated</span>")
        )
    return html


# ---------------- THEME 3: CHRONO WARP (超時空賽博神經網) ----------------

def _get_warp_html(lang: str = "zh") -> str:
    return """
    <style>
    @keyframes gridMove {
        0% { transform: translateY(0); }
        100% { transform: translateY(28px); }
    }
    @keyframes corePulse {
        0%, 100% { transform: scale(1) rotate(0deg); filter: drop-shadow(0 0 10px #7c4dff); }
        50% { transform: scale(1.18) rotate(180deg); filter: drop-shadow(0 0 25px #ec4899); }
    }
    .warp-canvas {
        width: 100%;
        height: 125px;
        position: relative;
        overflow: hidden;
        perspective: 380px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .warp-grid-floor {
        position: absolute;
        bottom: -35px;
        left: -50%;
        width: 200%;
        height: 110px;
        transform: rotateX(68deg);
        background-image:
            linear-gradient(rgba(124, 77, 255, 0.28) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124, 77, 255, 0.28) 1px, transparent 1px);
        background-size: 28px 28px;
        animation: gridMove 1.1s infinite linear;
    }
    .quantum-core {
        width: 46px;
        height: 46px;
        border: 2px solid #a855f7;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(124, 77, 255, 0.45), rgba(236, 72, 153, 0.45));
        animation: corePulse 2.8s infinite ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        z-index: 5;
    }
    .floating-code {
        position: absolute;
        font-family: monospace;
        font-size: 10.5px;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 2px 6px;
        border-radius: 4px;
        pointer-events: none;
    }
    </style>

    <div class="warp-canvas">
        <div class="warp-grid-floor"></div>

        <div class="quantum-core">
            <span>🔮</span>
        </div>

        <!-- Floating Temporal Code Snippets -->
        <div class="floating-code" style="top: 12px; left: 14%;">Transformer 330M</div>
        <div class="floating-code" style="bottom: 20px; left: 18%;">EpiWeek(t+h)</div>
        <div class="floating-code" style="top: 16px; right: 16%;">p=4, d=1, q=2</div>
        <div class="floating-code" style="bottom: 16px; right: 14%;">Fourier(S_12)</div>
    </div>
    """


# ---------------- THEME 4: COFFEE (防疫研究員的咖啡時光) ----------------

def _get_coffee_html(lang: str = "zh") -> str:
    html = """
    <style>
    @keyframes steamFloat {
        0% { transform: translateY(0) scaleX(1); opacity: 0.8; }
        50% { transform: translateY(-10px) scaleX(1.3); opacity: 0.4; }
        100% { transform: translateY(-20px) scaleX(1.6); opacity: 0; }
    }
    .coffee-scene {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 24px;
        width: 100%;
        height: 125px;
    }
    .cup-container {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .cup-steam-wrap {
        display: flex;
        gap: 7px;
        height: 22px;
    }
    .steam-line {
        width: 3px;
        height: 14px;
        background: linear-gradient(to top, rgba(254, 215, 170, 0.85), rgba(254, 215, 170, 0));
        border-radius: 3px;
        animation: steamFloat 1.8s infinite ease-out;
    }
    .cup-body {
        font-size: 38px;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));
    }
    .coffee-dialog {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(251, 146, 60, 0.4);
        padding: 9px 15px;
        border-radius: 12px;
        color: #fed7aa;
        font-size: 12px;
        line-height: 1.55;
        max-width: 320px;
    }
    </style>

    <div class="coffee-scene">
        <div class="cup-container">
            <div class="cup-steam-wrap">
                <div class="steam-line" style="animation-delay: 0s;"></div>
                <div class="steam-line" style="animation-delay: 0.4s;"></div>
                <div class="steam-line" style="animation-delay: 0.8s;"></div>
            </div>
            <div class="cup-body">☕</div>
        </div>

        <div class="coffee-dialog">
            <div style="font-weight: 700; color: #fb923c; margin-bottom: 2px;">☕ 防疫研究員日常提醒：</div>
            <div>數據正在慢火細燉，請稍候片刻。<br>喝口溫開水、舒展一下肩頸，精準預測馬上就端上桌！</div>
        </div>
    </div>
    """
    if lang == "en":
        html = (
            html.replace("☕ 防疫研究員日常提醒：", "☕ Epidemiologist's Daily Reminder:")
            .replace("數據正在慢火細燉，請稍候片刻。<br>喝口溫開水、舒展一下肩頸，精準預測馬上就端上桌！", "Data is simmering to perfection. Please hold on.<br>Grab a sip of water, stretch your neck and shoulders, accurate forecasts will be served shortly!")
        )
    return html
