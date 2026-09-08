"""
Streamlit Web Application:
Google TimesFM 3.0 vs. Meta Prophet vs. Auto ARIMA Epidemic 8-Period Forecasting Platform.
Features:
1. 3-Tier Model Comparison: Foundation Transformer (TimesFM 3.0) vs. Bayesian Additive (Prophet) vs. Classical (Auto ARIMA).
2. Taiwan CDC (疾管署) & MMWR EpiWeek mathematical rule engine (infinite year boundary coverage without static file expiration).
3. Right-Censored Reporting Lag Protection: Default option to exclude the latest incomplete/partial-week reporting data.
4. Multiplicative log-space epidemic adaptation for non-negative guarantees.
5. Strict Security Controls:
   - Zero-vulnerability dependency profile (verified by pip-audit).
   - Clean static analysis profile (verified by bandit with 0 High/Med/Low issues).
   - OWASP CSV Injection / DDE protection on exports.
   - Path traversal validation and size boundary enforcement.
   - HTML injection (XSS) escaping.
6. Mobile & Cross-Device Responsiveness:
   - Auto-wrapping metrics and summary cards via CSS flexbox.
   - Legend positioned safely below Plotly chart to prevent mobile title/curve overlapping.
   - Full-width responsive metric comparative bar charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import io
import html
import importlib
import utils.data_processor
importlib.reload(utils.data_processor)
import utils.animations
importlib.reload(utils.animations)
import utils.i18n
importlib.reload(utils.i18n)

from utils.i18n import (
    t,
    get_dataset_choices,
    get_freq_choices,
    get_eval_mode_choices,
    get_time_window_choices,
    get_column_display_name
)
from utils.animations import (
    get_animation_themes,
    render_loading_card,
    get_random_quote
)
from utils.data_processor import (
    load_dataset,
    detect_columns,
    prepare_epidemic_data,
    generate_future_dates,
    future_dates_to_year_weeks,
    sanitize_dataframe_for_csv_export,
    is_safe_url,
    fetch_live_csv_data
)
from utils.metrics import compute_forecast_metrics
from models.prophet_wrapper import ProphetForecasterWrapper
from models.timesfm_wrapper import TimesFMWrapper
from models.arima_wrapper import AutoARIMAForecasterWrapper

# Page Config
st.set_page_config(
    page_title="TimesFM 3.0 vs. Prophet vs. Auto ARIMA Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (With full Mobile Responsiveness & Overflow Protection)
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #42A5F5 0%, #AB47BC 35%, #66BB6A 70%, #EF5350 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        word-break: break-word;
    }
    .sub-header {
        font-size: 0.95rem;
        color: var(--text-color, #64748b);
        margin-bottom: 1.0rem;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .metric-card-tfm, .metric-card-prophet, .metric-card-arima {
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        color: var(--text-color, inherit) !important;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 15px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
        word-break: break-word;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card-tfm {
        border-left: 5px solid #1E88E5 !important;
    }
    .metric-card-prophet {
        border-left: 5px solid #FF9800 !important;
    }
    .metric-card-arima {
        border-left: 5px solid #2E7D32 !important;
    }
    .metric-card-tfm h4, .metric-card-prophet h4, .metric-card-arima h4 {
        color: var(--text-color, inherit) !important;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .metric-card-tfm p, .metric-card-prophet p, .metric-card-arima p {
        color: var(--text-color, inherit) !important;
        margin-bottom: 6px;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .metric-card-tfm b, .metric-card-prophet b, .metric-card-arima b {
        color: var(--text-color, inherit) !important;
    }

    /* System & Theme Dark Mode Explicit Overrides */
    @media (prefers-color-scheme: dark) {
        .metric-card-tfm, .metric-card-prophet, .metric-card-arima {
            background-color: #1e2530 !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
            color: #f1f5f9 !important;
        }
        .metric-card-tfm h4, .metric-card-prophet h4, .metric-card-arima h4,
        .metric-card-tfm p, .metric-card-prophet p, .metric-card-arima p,
        .metric-card-tfm b, .metric-card-prophet b, .metric-card-arima b {
            color: #f1f5f9 !important;
        }
    }
    .tag-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .tag-blue { background-color: #E3F2FD; color: #1565C0; }
    .tag-orange { background-color: #FFF3E0; color: #E65100; }
    .tag-green { background-color: #E8F5E9; color: #2E7D32; }
    .tag-purple { background-color: #F3E5F5; color: #7B1FA2; }

    /* Mobile & Small Screen Responsive Overrides */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.35rem !important;
            line-height: 1.35 !important;
        }
        .sub-header {
            font-size: 0.82rem !important;
            gap: 4px !important;
        }
        .tag-badge {
            font-size: 0.72rem !important;
            padding: 2px 7px !important;
        }
        /* Mobile: Columns wrap into neat 2-per-row grid */
        div[data-testid="column"] {
            min-width: 45% !important;
            flex: 1 1 45% !important;
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
        /* Summary cards take 100% width on phone */
        .metric-card-tfm, .metric-card-prophet, .metric-card-arima {
            min-width: 100% !important;
            padding: 12px 14px !important;
        }
        /* Tab list: smooth horizontal touch scrolling without wrapping */
        div[data-baseweb="tab-list"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            white-space: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
            padding-bottom: 4px !important;
        }
        div[data-baseweb="tab"] {
            font-size: 0.82rem !important;
            padding: 6px 10px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_live_csv(url: str, fallback_path: str = None):
    """Cached fetcher for external live OpenData / CSV URLs (TTL 1 hour)."""
    return fetch_live_csv_data(url, timeout=12, fallback_path=fallback_path)


@st.cache_resource(show_spinner=False)
def get_timesfm_model():
    """Cached singleton for Google TimesFM 3.0 model."""
    return TimesFMWrapper(device='cpu')


def get_prophet_model(
    interval_width: float = 0.8,
    use_log_transform: bool = True,
    changepoint_prior_scale: float = 0.15,
    changepoint_range: float = 0.95
):
    """Factory for optimized Meta Prophet model."""
    return ProphetForecasterWrapper(
        interval_width=interval_width,
        use_log_transform=use_log_transform,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=changepoint_range
    )


def get_arima_model(
    interval_width: float = 0.8,
    use_log_transform: bool = True,
    max_p: int = 4,
    max_q: int = 4,
    seasonal: bool = False
):
    """Factory for Auto ARIMA model."""
    return AutoARIMAForecasterWrapper(
        interval_width=interval_width,
        use_log_transform=use_log_transform,
        max_p=max_p,
        max_q=max_q,
        seasonal=seasonal
    )


# ---------------- SIDEBAR CONTROLS ----------------
# Language Switcher at Top of Sidebar
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'zh'

lang_options = ["繁體中文", "English"]
curr_lang_idx = 0 if st.session_state['lang'] == 'zh' else 1
selected_lang_label = st.sidebar.radio(
    t("language_selector_label", st.session_state.get('lang', 'zh')),
    lang_options,
    index=curr_lang_idx,
    horizontal=True,
    key="lang_switcher_radio"
)
lang = 'zh' if selected_lang_label == "繁體中文" else 'en'
st.session_state['lang'] = lang

st.sidebar.markdown(t("sidebar_header", lang))

# Dataset selection using persistent stable keys
if 'selected_ds_key' not in st.session_state:
    st.session_state['selected_ds_key'] = 'enterovirus'

ds_choices = get_dataset_choices(lang)
ds_keys = list(ds_choices.keys())
ds_labels = [ds_choices[k] for k in ds_keys]

cur_ds_idx = ds_keys.index(st.session_state['selected_ds_key']) if st.session_state['selected_ds_key'] in ds_keys else 0

selected_ds_label = st.sidebar.radio(
    t("data_source_label", lang),
    ds_labels,
    index=cur_ds_idx
)
selected_ds_key = ds_keys[ds_labels.index(selected_ds_label)]
st.session_state['selected_ds_key'] = selected_ds_key

sample_map = {
    "enterovirus": "sample_data/enterovirus_weekly.csv",
    "dengue": "sample_data/dengue_weekly.csv",
    "scrub_typhus": "sample_data/scrub_typhus_monthly.csv",
    "covid": "sample_data/covid19_daily.csv",
    "flu": "sample_data/flu_weekly.csv"
}

raw_df = None

if selected_ds_key in sample_map:
    file_path = sample_map[selected_ds_key]
    if os.path.exists(file_path):
        raw_df = load_dataset(file_path)
elif selected_ds_key == "cbc_forex":
    cbc_url = "https://www.cbc.gov.tw/public/data/OpenData/%E5%A4%96%E5%8C%AF%E5%B1%80/FTDOpenData015.csv"
    fallback_file = "sample_data/ntd_usd_daily.csv"
    st.sidebar.markdown(t("cbc_header", lang))
    st.sidebar.caption(t("cbc_caption", lang))
    with st.sidebar.status(t("cbc_status_connecting", lang), expanded=False) as status:
        try:
            raw_df, is_live, fetch_msg = get_cached_live_csv(cbc_url, fallback_path=fallback_file)
            status.update(label=t("cbc_status_complete", lang), state="complete")
        except Exception as e:
            status.update(label=t("cbc_status_error", lang), state="error")
            st.sidebar.error(f"{t('load_failed', lang)}: {html.escape(str(e))}")
    if raw_df is not None:
        if is_live:
            st.sidebar.success(t("cbc_success_live", lang, count=f"{len(raw_df):,}"))
        else:
            st.sidebar.info(t("cbc_fallback_info", lang, msg=fetch_msg, count=f"{len(raw_df):,}"))
            with st.sidebar.expander(t("cbc_expander_title", lang), expanded=False):
                st.caption(t("cbc_expander_desc", lang))
        if st.sidebar.button(t("cbc_resync_button", lang), help=t("cbc_resync_help", lang)):
            st.cache_data.clear()
            st.rerun()
elif selected_ds_key == "custom_url":
    st.sidebar.markdown(t("custom_url_header", lang))
    custom_url = st.sidebar.text_input(
        t("custom_url_label", lang),
        value="https://www.cbc.gov.tw/public/data/OpenData/%E5%A4%96%E5%8C%AF%E5%B1%80/FTDOpenData015.csv",
        help=t("custom_url_help", lang)
    )
    if custom_url.strip():
        with st.sidebar.status(t("custom_url_status_loading", lang), expanded=False) as status:
            try:
                # Provide local fallback if default CBC URL
                fb = "sample_data/ntd_usd_daily.csv" if "FTDOpenData015" in custom_url else None
                raw_df, is_live, fetch_msg = get_cached_live_csv(custom_url.strip(), fallback_path=fb)
                status.update(label=t("custom_url_status_complete", lang), state="complete")
            except Exception as e:
                status.update(label=t("custom_url_status_error", lang), state="error")
                st.sidebar.error(f"{t('custom_url_parse_failed', lang)}: {html.escape(str(e))}")
        if raw_df is not None:
            if is_live:
                st.sidebar.success(t("custom_url_success", lang, count=f"{len(raw_df):,}"))
            else:
                st.sidebar.info(t("custom_url_fallback_info", lang, msg=fetch_msg, count=f"{len(raw_df):,}"))
            if st.sidebar.button(t("custom_url_resync_button", lang), help=t("custom_url_resync_help", lang)):
                st.cache_data.clear()
                st.rerun()
elif selected_ds_key == "upload":
    uploaded_file = st.sidebar.file_uploader(t("upload_label", lang), type=['csv', 'xlsx', 'xls', 'json'])
    if uploaded_file is not None:
        try:
            raw_df = load_dataset(uploaded_file)
            st.sidebar.success(t("upload_success", lang, name=html.escape(uploaded_file.name), count=f"{len(raw_df):,}"))
        except Exception as e:
            st.sidebar.error(f"{t('parse_failed', lang)}: {html.escape(str(e))}")
elif selected_ds_key == "paste":
    default_text = """"發病年週","確定病例數"
"202434",3
"202435",19
"202436",48
"202437",43
"202438",44
"202439",37
"202440",26
"202441",31
"202442",26
"202443",25
"202444",23
"202445",17
"202446",13
"202447",9
"202448",18
"202449",11
"202450",10
"202451",14
"202452",10
"202501",12
"202502",11
"202503",8
"202504",11
"202505",4
"202506",9
"202507",13
"202508",12
"202509",5
"202510",7
"202511",7
"202512",6
"202513",10
"202514",11
"202515",20
"202516",17
"202517",15
"202518",39
"202519",57
"202520",70
"202521",119
"202522",137
"202523",149
"202524",153
"202525",118
"202526",141
"202527",127
"202528",99
"202529",59
"202530",51
"202531",39
"202532",28
"202533",25
"202534",23
"202535",12
"202536",15
"202537",20
"202538",5
"202539",7
"202540",9
"202541",8
"202542",2
"202543",6
"202544",6
"202545",2
"202546",6
"202547",2
"202548",3
"202549",3
"202550",1
"202551",3
"202552",3
"202553",2
"202601",3
"202602",1
"202603",4
"202604",3
"202605",0
"202606",3
"202607",2
"202608",0
"202609",4
"202610",2
"202611",4
"202612",1
"202613",2
"202614",1
"202615",0
"202616",0
"202617",1
"202618",0
"202619",3
"202620",0
"202621",2
"202622",3
"202623",5
"202624",3
"202625",11
"202626",15
"202627",15
"202628",32
"202629",44
"202630",60
"202631",79
"202632",80
"202633",62
"202634",63
"202635",14"""
    pasted_csv = st.sidebar.text_area(t("paste_label", lang), default_text, height=180)
    if pasted_csv.strip():
        try:
            raw_df = load_dataset(io.StringIO(pasted_csv))
        except Exception as e:
            st.sidebar.error(f"{t('csv_parse_failed', lang)}: {html.escape(str(e))}")

if raw_df is None or len(raw_df) == 0:
    st.error(t("no_data_error", lang))
    st.stop()

# Detect Columns
auto_date, auto_target, all_cols = detect_columns(raw_df)

col_date = st.sidebar.selectbox(
    t("col_date_label", lang),
    all_cols,
    index=all_cols.index(auto_date) if auto_date in all_cols else 0,
    format_func=lambda c: get_column_display_name(c, lang)
)
remaining_cols = [c for c in all_cols if c != col_date]
target_idx = remaining_cols.index(auto_target) if auto_target in remaining_cols else 0
col_target = st.sidebar.selectbox(
    t("col_target_label", lang),
    remaining_cols,
    index=target_idx if remaining_cols else 0,
    format_func=lambda c: get_column_display_name(c, lang)
)

target_disp = get_column_display_name(col_target, lang)
date_col_disp = get_column_display_name(col_date, lang)

# Preprocess Data (with exact CDC EpiWeek mathematical engine)
clean_df, data_stats = prepare_epidemic_data(raw_df, col_date, col_target)

# Preserve complete multi-year dataframe for historical overlay analysis
full_history_df = clean_df.copy()

# Optional Time Window Slicing (Resolves baseline flattening from extreme historic outbreaks)
if len(clean_df) > 30 and (clean_df['ds'].max() - clean_df['ds'].min()).days > 500:
    min_year = int(clean_df['ds'].min().year)
    max_year = int(clean_df['ds'].max().year)
    if max_year - min_year >= 2:
        time_windows = get_time_window_choices(min_year, max_year, lang)
        window_ids = [w[0] for w in time_windows]
        window_labels = [w[1] for w in time_windows]

        if 'selected_window_id' not in st.session_state:
            st.session_state['selected_window_id'] = 2024 if 2024 in window_ids else 'all'

        cur_w_idx = window_ids.index(st.session_state['selected_window_id']) if st.session_state['selected_window_id'] in window_ids else 0
        sel_w_label = st.sidebar.selectbox(
            t("time_window_label", lang),
            window_labels,
            index=cur_w_idx,
            help=t("time_window_help", lang)
        )
        selected_wid = window_ids[window_labels.index(sel_w_label)]
        st.session_state['selected_window_id'] = selected_wid

        if selected_wid != 'all':
            start_yr = int(selected_wid)
            clean_df = clean_df[clean_df['ds'] >= f"{start_yr}-01-01"].reset_index(drop=True)
            data_stats['count'] = len(clean_df)
            data_stats['min'] = float(clean_df['y'].min())
            data_stats['max'] = float(clean_df['y'].max())
            data_stats['mean'] = float(clean_df['y'].mean())
            data_stats['start_date'] = clean_df['ds'].min()
            if 'year_week' in clean_df and clean_df['year_week'].iloc[0] is not None:
                data_stats['start_year_week'] = clean_df['year_week'].iloc[0]

# Horizon & Frequency configuration
st.sidebar.markdown("---")
st.sidebar.markdown(t("forecast_params_header", lang))

horizon = st.sidebar.slider(
    t("horizon_label", lang),
    min_value=4,
    max_value=24,
    value=8,
    step=1,
    help=t("horizon_help", lang)
)

freq_choices = get_freq_choices(lang)
freq_keys = list(freq_choices.keys())
freq_labels = [freq_choices[k] for k in freq_keys]

if 'selected_freq_key' not in st.session_state:
    st.session_state['selected_freq_key'] = data_stats.get('frequency', 'W')

cur_freq_idx = freq_keys.index(st.session_state['selected_freq_key']) if st.session_state['selected_freq_key'] in freq_keys else 0
sel_freq_label = st.sidebar.selectbox(
    t("frequency_label", lang),
    freq_labels,
    index=cur_freq_idx
)
selected_freq = freq_keys[freq_labels.index(sel_freq_label)]
st.session_state['selected_freq_key'] = selected_freq

is_monthly_series = (selected_freq == 'M') or data_stats.get('is_year_month_converted', False)
is_weekly_series = (selected_freq == 'W') or data_stats.get('is_year_week_converted', False)
if is_monthly_series:
    unit_label = t("unit_month", lang)
elif is_weekly_series:
    unit_label = t("unit_week", lang)
else:
    unit_label = t("unit_day", lang)

eval_choices = get_eval_mode_choices(lang)
eval_keys = list(eval_choices.keys())
eval_labels = [eval_choices[k] for k in eval_keys]

if 'selected_eval_mode' not in st.session_state:
    st.session_state['selected_eval_mode'] = 'future'

cur_eval_idx = eval_keys.index(st.session_state['selected_eval_mode']) if st.session_state['selected_eval_mode'] in eval_keys else 0
sel_eval_label = st.sidebar.radio(
    t("eval_mode_label", lang),
    eval_labels,
    index=cur_eval_idx
)
selected_eval = eval_keys[eval_labels.index(sel_eval_label)]
st.session_state['selected_eval_mode'] = selected_eval
is_backtesting = (selected_eval == 'backtest')

# Option: Default exclude incomplete latest week
default_exclude = bool(
    (data_stats.get('is_year_week_converted', False) or data_stats.get('is_year_month_converted', False))
    and data_stats.get('frequency') != 'D'
)
exclude_incomplete_last = st.sidebar.checkbox(
    t("exclude_last_label", lang),
    value=default_exclude,
    help=t("exclude_last_help", lang)
)

confidence_level = st.sidebar.selectbox(
    t("confidence_interval_label", lang),
    [0.80, 0.90, 0.95],
    index=0,
    format_func=lambda x: t(
        "ci_format",
        lang,
        pct=int(x * 100),
        low=int((1 - x) / 2 * 100),
        high=int((1 - (1 - x) / 2) * 100)
    )
)

st.sidebar.markdown("---")
st.sidebar.markdown(t("anim_section_header", lang))

anim_themes = get_animation_themes(lang)
anim_keys = list(anim_themes.keys())
anim_labels = [anim_themes[k] for k in anim_keys]

if 'selected_anim_key' not in st.session_state:
    st.session_state['selected_anim_key'] = 'random'

cur_anim_idx = anim_keys.index(st.session_state['selected_anim_key']) if st.session_state['selected_anim_key'] in anim_keys else 0
sel_anim_label = st.sidebar.selectbox(
    t("anim_theme_label", lang),
    anim_labels,
    index=cur_anim_idx,
    help=t("anim_theme_help", lang)
)
anim_theme_key = anim_keys[anim_labels.index(sel_anim_label)]
st.session_state['selected_anim_key'] = anim_theme_key

if st.sidebar.button(t("anim_preview_button", lang), key="btn_preview_anim_toggle", help=t("anim_preview_help", lang)):
    st.session_state['show_anim_preview'] = True

# Model Parameters Tuning
with st.sidebar.expander(t("prophet_expander", lang), expanded=False):
    prophet_log = st.checkbox(t("prophet_log_label", lang), value=True, help=t("prophet_log_help", lang))
    prophet_cp_scale = st.slider(t("prophet_cp_scale_label", lang), min_value=0.01, max_value=0.50, value=0.15, step=0.02, help=t("prophet_cp_scale_help", lang))
    prophet_cp_range = st.slider(t("prophet_cp_range_label", lang), min_value=0.80, max_value=0.98, value=0.95, step=0.01, help=t("prophet_cp_range_help", lang))

with st.sidebar.expander(t("arima_expander", lang), expanded=False):
    arima_log = st.checkbox(t("arima_log_label", lang), value=True, help=t("arima_log_help", lang))
    arima_max_p = st.number_input(t("arima_max_p_label", lang), min_value=1, max_value=8, value=4)
    arima_max_q = st.number_input(t("arima_max_q_label", lang), min_value=1, max_value=8, value=4)
    arima_seasonal = st.checkbox(t("arima_seasonal_label", lang), value=(selected_freq == 'D'), help=t("arima_seasonal_help", lang))

# Allow restoring dismissed notices in sidebar
if st.session_state.get('dismiss_system_notices', False):
    st.sidebar.markdown("---")
    if st.sidebar.button(t("restore_notices_btn", lang), help=t("restore_notices_help", lang)):
        st.session_state['dismiss_system_notices'] = False
        st.rerun()

# ---------------- MAIN CONTENT ----------------
st.markdown(f'<div class="main-header">{t("main_header", lang)}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    f'<span class="tag-badge tag-blue">{t("badge_tfm", lang)}</span>'
    f'<span class="tag-badge tag-orange">{t("badge_prophet", lang)}</span>'
    f'<span class="tag-badge tag-green">{t("badge_arima", lang)}</span>'
    f'<span class="tag-badge tag-purple">{t("badge_engine", lang)}</span>'
    '</div>',
    unsafe_allow_html=True
)

# Handle Request: Exclude incomplete latest period
has_excluded_period = False
excluded_record = None
ex_label = ""
last_full_label = ""
last_full_val = 0.0

if exclude_incomplete_last and len(clean_df) > 8:
    excluded_record = clean_df.iloc[-1].copy()
    base_working_df = clean_df.iloc[:-1].copy()
    has_excluded_period = True
    ex_label = excluded_record['year_month'] if data_stats.get('is_year_month_converted') else (excluded_record['year_week'] if data_stats['is_year_week_converted'] else excluded_record['ds'].strftime('%Y-%m-%d'))
    last_full_label = base_working_df['year_month'].iloc[-1] if data_stats.get('is_year_month_converted') else (base_working_df['year_week'].iloc[-1] if data_stats['is_year_week_converted'] else base_working_df['ds'].iloc[-1].strftime('%Y-%m-%d'))
    last_full_val = base_working_df['y'].iloc[-1]
else:
    base_working_df = clean_df.copy()
    has_excluded_period = False

# Manage Notices Expander & Dismiss State
if 'dismiss_system_notices' not in st.session_state:
    st.session_state['dismiss_system_notices'] = False

has_any_notice = data_stats['is_year_week_converted'] or data_stats.get('is_year_month_converted') or has_excluded_period

if has_any_notice and not st.session_state['dismiss_system_notices']:
    summary_chips = []
    if data_stats.get('is_year_month_converted'):
        summary_chips.append(f"📅 {t('chip_ym', lang, start=data_stats['start_year_month'], end=data_stats['end_year_month'])}")
    elif data_stats['is_year_week_converted']:
        summary_chips.append(f"📅 {t('chip_yw', lang, start=data_stats['start_year_week'], end=data_stats['end_year_week'])}")

    if has_excluded_period and excluded_record is not None:
        unit_str = t("unit_month_incomplete", lang) if data_stats.get('is_year_month_converted') else t("unit_week_incomplete", lang)
        unit_cases = t("unit_cases", lang)
        ex_val_str = f"{excluded_record['y']:,.0f}"
        summary_chips.append(f"🛡️ {t('chip_excluded', lang, unit=unit_str, label=ex_label, val=ex_val_str, cases=unit_cases)}")

    chip_desc = " ｜ ".join(summary_chips)

    with st.expander(f"ℹ️ {t('notices_expander_title', lang)} ({chip_desc})", expanded=False):
        if data_stats.get('is_year_month_converted'):
            safe_col = html.escape(str(date_col_disp))
            st.info(
                f"📅 **{t('info_ym_title', lang)}** "
                f"{t('info_ym_body', lang, col=safe_col, start=data_stats['start_year_month'], end=data_stats['end_year_month'], count=len(clean_df))}"
            )
        elif data_stats['is_year_week_converted']:
            safe_col = html.escape(str(date_col_disp))
            st.info(
                f"📅 **{t('info_yw_title', lang)}** "
                f"{t('info_yw_body', lang, col=safe_col, start=data_stats['start_year_week'], end=data_stats['end_year_week'])}"
            )
        if has_excluded_period and excluded_record is not None:
            unit_str = t("unit_month_incomplete", lang) if data_stats.get('is_year_month_converted') else t("unit_week_incomplete", lang)
            unit_cases = t("unit_cases", lang)
            ex_val_str = f"{excluded_record['y']:,.0f}"
            last_val_str = f"{last_full_val:,.0f}"
            st.warning(
                f"🛡️ **{t('warn_excluded_title', lang, unit=unit_str)}** "
                f"{t('warn_excluded_body', lang, unit=unit_str, label=ex_label, val=ex_val_str, cases=unit_cases, last_label=last_full_label, last_val=last_val_str)}"
            )
        col_sp, col_dismiss = st.columns([0.78, 0.22])
        with col_dismiss:
            if st.button(t("btn_dismiss_notices", lang), key="btn_dismiss_banner", help=t("btn_dismiss_help", lang), use_container_width=True):
                st.session_state['dismiss_system_notices'] = True
                st.rerun()

# Optional animation preview block
if st.session_state.get('show_anim_preview', False):
    c_prev1, c_prev2 = st.columns([0.8, 0.2])
    with c_prev1:
        st.info(f"💡 **{t('anim_preview_title', lang)}**")
    with c_prev2:
        if st.button(t("btn_close_preview", lang), key="btn_close_anim_preview", use_container_width=True):
            st.session_state['show_anim_preview'] = False
            st.rerun()
    st.html(
        render_loading_card(
            theme_key=anim_theme_key,
            step_num=2,
            step_desc=t("anim_preview_step_desc", lang, theme=anim_themes.get(anim_theme_key, '')),
            lang=lang
        )
    )

# Dataset overview cards
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
unit_period = t("unit_periods", lang)
unit_str = t("unit_month_incomplete", lang) if data_stats.get('is_year_month_converted') else t("unit_week_incomplete", lang)

with col_m1:
    st.metric(
        t("metric_model_steps", lang),
        f"{len(base_working_df)} {unit_period}",
        help=t("metric_model_steps_help", lang, total=len(clean_df), ex_count=1 if has_excluded_period else 0, unit=unit_str, periods=unit_period)
    )
with col_m2:
    disp_val = base_working_df['y'].iloc[-1]
    disp_yw = base_working_df['year_month'].iloc[-1] if data_stats.get('is_year_month_converted') else (base_working_df['year_week'].iloc[-1] if data_stats['is_year_week_converted'] else base_working_df['ds'].iloc[-1].strftime('%Y-%m-%d'))
    st.metric(
        t("metric_baseline_val", lang),
        f"{disp_val:,.0f}",
        help=t("metric_baseline_help", lang, yw=disp_yw)
    )
with col_m3:
    st.metric(
        t("metric_peak_val", lang),
        f"{data_stats['peak_val']:,.0f}",
        help=t("metric_peak_help", lang, date=data_stats['peak_date'].strftime('%Y-%m-%d'))
    )
with col_m4:
    freq_display_map = {'W': t("freq_w", lang), 'D': t("freq_d", lang), 'M': t("freq_m", lang)}
    st.metric(t("metric_frequency", lang), freq_display_map.get(selected_freq, selected_freq))
with col_m5:
    st.metric(t("metric_recent_trend", lang), f"{data_stats['recent_trend_pct']:+.1f}%")

st.markdown("---")

# Prepare train/test split if backtesting
if is_backtesting:
    if len(base_working_df) <= horizon + 4:
        st.error(t("error_insufficient_data", lang, req=horizon + 4, horizon=horizon))
        st.stop()
    train_df = base_working_df.iloc[:-horizon].copy()
    ground_truth_df = base_working_df.iloc[-horizon:].copy()
    future_dates_list = [d.strftime('%Y-%m-%d') for d in ground_truth_df['ds']]
    if data_stats.get('is_year_month_converted'):
        future_yws_list = list(ground_truth_df['year_month'])
    else:
        future_yws_list = list(ground_truth_df['year_week'])
    last_hist_date = train_df['ds'].iloc[-1]
    last_hist_val = train_df['y'].iloc[-1]
else:
    train_df = base_working_df.copy()
    ground_truth_df = None
    last_hist_date = train_df['ds'].iloc[-1]
    last_hist_val = train_df['y'].iloc[-1]
    future_dates_dt = generate_future_dates(last_hist_date, horizon=horizon, freq=selected_freq)
    future_dates_list = [d.strftime('%Y-%m-%d') for d in future_dates_dt]
    if data_stats.get('is_year_month_converted'):
        future_yws_list = [d.strftime('%Y%m') for d in future_dates_dt]
    else:
        future_yws_list = future_dates_to_year_weeks(future_dates_dt)


# Execution with dynamic transition animation
anim_placeholder = st.empty()
chosen_quote = get_random_quote(lang=lang)

# Phase 1: Google TimesFM 3.0
anim_placeholder.html(
    render_loading_card(
        theme_key=anim_theme_key,
        step_num=1,
        step_desc=t("anim_step_1", lang),
        quote=chosen_quote,
        lang=lang
    )
)

# Load Models
tfm_model = get_timesfm_model()
prophet_model = get_prophet_model(
    interval_width=confidence_level,
    use_log_transform=prophet_log,
    changepoint_prior_scale=prophet_cp_scale,
    changepoint_range=prophet_cp_range
)
arima_model = get_arima_model(
    interval_width=confidence_level,
    use_log_transform=arima_log,
    max_p=int(arima_max_p),
    max_q=int(arima_max_q),
    seasonal=arima_seasonal
)

# 1. TimesFM 3.0 forecast
res_tfm = tfm_model.forecast(
    df_history=train_df,
    future_dates=future_dates_list,
    horizon=horizon,
    freq=selected_freq,
    confidence_level=confidence_level
)

# Phase 2: Meta Prophet
anim_placeholder.html(
    render_loading_card(
        theme_key=anim_theme_key,
        step_num=2,
        step_desc=t("anim_step_2", lang),
        quote=chosen_quote,
        lang=lang
    )
)

# 2. Prophet forecast
res_prophet = prophet_model.forecast(
    df_history=train_df,
    horizon=horizon,
    freq=selected_freq
)

# Phase 3: Auto ARIMA
anim_placeholder.html(
    render_loading_card(
        theme_key=anim_theme_key,
        step_num=3,
        step_desc=t("anim_step_3", lang),
        quote=chosen_quote,
        lang=lang
    )
)

# 3. Auto ARIMA forecast
res_arima = arima_model.forecast(
    df_history=train_df,
    future_dates=future_dates_list,
    horizon=horizon,
    freq=selected_freq
)

# Clear animation container once inference completes
anim_placeholder.empty()

# Compute metrics if backtesting
metrics_tfm = {}
metrics_prophet = {}
metrics_arima = {}
if is_backtesting and ground_truth_df is not None:
    actual_vals = ground_truth_df['y'].values
    metrics_tfm = compute_forecast_metrics(
        actual=actual_vals,
        predicted=res_tfm['forecast'],
        last_historical_val=last_hist_val,
        ci_lower=res_tfm['ci_lower'],
        ci_upper=res_tfm['ci_upper']
    )
    metrics_prophet = compute_forecast_metrics(
        actual=actual_vals,
        predicted=res_prophet['forecast'],
        last_historical_val=last_hist_val,
        ci_lower=res_prophet['ci_lower'],
        ci_upper=res_prophet['ci_upper']
    )
    metrics_arima = compute_forecast_metrics(
        actual=actual_vals,
        predicted=res_arima['forecast'],
        last_historical_val=last_hist_val,
        ci_lower=res_arima['ci_lower'],
        ci_upper=res_arima['ci_upper']
    )


# ---------------- DASHBOARD TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    t("tab1_title", lang),
    t("tab2_title", lang),
    t("tab3_title", lang),
    t("tab4_title", lang)
])

with tab1:
    c_tab_head, c_scale_sel = st.columns([0.58, 0.42])
    with c_tab_head:
        st.markdown(f"### 📈 {t('tab1_header', lang, target=html.escape(str(target_disp)), horizon=horizon)}")
    with c_scale_sel:
        chart_y_scale = st.radio(
            t("scale_mode_label", lang),
            [t("scale_linear", lang), t("scale_log", lang)],
            index=0,
            horizontal=True,
            help=t("scale_mode_help", lang)
        )
    is_log_scale = chart_y_scale == t("scale_log", lang)

    def transform_y(vals):
        if not is_log_scale:
            return vals
        return [max(float(v), 0.2) if (v is not None and not np.isnan(v)) else v for v in vals]

    # Plotly Interactive Chart
    fig = go.Figure()

    hist_dates = [d.strftime('%Y-%m-%d') for d in train_df['ds']]
    hist_vals = train_df['y'].values
    hist_yws = list(train_df['year_week'])

    # Connection lines data
    all_tfm_dates = [hist_dates[-1]] + future_dates_list
    all_tfm_vals = [hist_vals[-1]] + list(res_tfm['forecast'])
    all_tfm_yws = [hist_yws[-1]] + future_yws_list
    all_tfm_lower = [hist_vals[-1]] + list(res_tfm['ci_lower'])
    all_tfm_upper = [hist_vals[-1]] + list(res_tfm['ci_upper'])

    all_prophet_dates = [hist_dates[-1]] + future_dates_list
    all_prophet_vals = [hist_vals[-1]] + list(res_prophet['forecast'])
    all_prophet_yws = [hist_yws[-1]] + future_yws_list
    all_prophet_lower = [hist_vals[-1]] + list(res_prophet['ci_lower'])
    all_prophet_upper = [hist_vals[-1]] + list(res_prophet['ci_upper'])

    all_arima_dates = [hist_dates[-1]] + future_dates_list
    all_arima_vals = [hist_vals[-1]] + list(res_arima['forecast'])
    all_arima_yws = [hist_yws[-1]] + future_yws_list
    all_arima_lower = [hist_vals[-1]] + list(res_arima['ci_lower'])
    all_arima_upper = [hist_vals[-1]] + list(res_arima['ci_upper'])

    # Hover templates based on whether Year-Month or Year-Week is available
    if data_stats.get('is_year_month_converted'):
        cd_hist = np.stack([hist_yws, hist_vals], axis=-1)
        cd_tfm = np.stack([all_tfm_yws, all_tfm_vals], axis=-1)
        cd_pro = np.stack([all_prophet_yws, all_prophet_vals], axis=-1)
        cd_ari = np.stack([all_arima_yws, all_arima_vals], axis=-1)
        hist_hover = t("hover_hist_ym", lang)
        tfm_hover = t("hover_tfm_ym", lang)
        pro_hover = t("hover_pro_ym", lang)
        ari_hover = t("hover_ari_ym", lang)
        gt_hover = t("hover_gt_ym", lang)
    elif data_stats['is_year_week_converted']:
        cd_hist = np.stack([hist_yws, hist_vals], axis=-1)
        cd_tfm = np.stack([all_tfm_yws, all_tfm_vals], axis=-1)
        cd_pro = np.stack([all_prophet_yws, all_prophet_vals], axis=-1)
        cd_ari = np.stack([all_arima_yws, all_arima_vals], axis=-1)
        hist_hover = t("hover_hist_yw", lang)
        tfm_hover = t("hover_tfm_yw", lang)
        pro_hover = t("hover_pro_yw", lang)
        ari_hover = t("hover_ari_yw", lang)
        gt_hover = t("hover_gt_yw", lang)
    else:
        cd_hist = np.stack([hist_dates, hist_vals], axis=-1)
        cd_tfm = np.stack([all_tfm_dates, all_tfm_vals], axis=-1)
        cd_pro = np.stack([all_prophet_dates, all_prophet_vals], axis=-1)
        cd_ari = np.stack([all_arima_dates, all_arima_vals], axis=-1)
        hist_hover = t("hover_hist_d", lang)
        tfm_hover = t("hover_tfm_d", lang)
        pro_hover = t("hover_pro_d", lang)
        ari_hover = t("hover_ari_d", lang)
        gt_hover = t("hover_gt_d", lang)

    # Historical Series
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=transform_y(hist_vals),
        customdata=cd_hist,
        mode='lines+markers',
        name=t("trace_history", lang),
        line=dict(color='#78909C', width=2.5),
        marker=dict(size=4, color='#78909C'),
        hovertemplate=hist_hover
    ))

    # Confidence Bands
    fig.add_trace(go.Scatter(
        x=all_tfm_dates + all_tfm_dates[::-1],
        y=transform_y(all_tfm_upper + all_tfm_lower[::-1]),
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        legendgroup="tfm",
        name=t("trace_tfm_ci", lang)
    ))

    fig.add_trace(go.Scatter(
        x=all_prophet_dates + all_prophet_dates[::-1],
        y=transform_y(all_prophet_upper + all_prophet_lower[::-1]),
        fill='toself',
        fillcolor='rgba(255, 152, 0, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        legendgroup="prophet",
        name=t("trace_prophet_ci", lang)
    ))

    fig.add_trace(go.Scatter(
        x=all_arima_dates + all_arima_dates[::-1],
        y=transform_y(all_arima_upper + all_arima_lower[::-1]),
        fill='toself',
        fillcolor='rgba(46, 125, 50, 0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        legendgroup="arima",
        name=t("trace_arima_ci", lang)
    ))

    # Forecast Lines
    fig.add_trace(go.Scatter(
        x=all_tfm_dates,
        y=transform_y(all_tfm_vals),
        customdata=cd_tfm,
        mode='lines+markers',
        name=t("trace_tfm_pred", lang),
        legendgroup="tfm",
        line=dict(color='#1E88E5', width=3.5),
        marker=dict(size=7, color='#1E88E5', symbol='circle'),
        hovertemplate=tfm_hover
    ))

    fig.add_trace(go.Scatter(
        x=all_prophet_dates,
        y=transform_y(all_prophet_vals),
        customdata=cd_pro,
        mode='lines+markers',
        name=t("trace_prophet_pred", lang),
        legendgroup="prophet",
        line=dict(color='#FF9800', width=3.0, dash='dash'),
        marker=dict(size=7, color='#FF9800', symbol='diamond'),
        hovertemplate=pro_hover
    ))

    fig.add_trace(go.Scatter(
        x=all_arima_dates,
        y=transform_y(all_arima_vals),
        customdata=cd_ari,
        mode='lines+markers',
        name=t("trace_arima_pred", lang),
        legendgroup="arima",
        line=dict(color='#2E7D32', width=2.8, dash='dashdot'),
        marker=dict(size=7, color='#2E7D32', symbol='triangle-up'),
        hovertemplate=ari_hover
    ))

    # Plot hollow marker for the excluded incomplete point
    if has_excluded_period and excluded_record is not None:
        ex_d_str = excluded_record['ds'].strftime('%Y-%m-%d')
        ex_v = excluded_record['y']
        ex_w = excluded_record['year_week'] if data_stats['is_year_week_converted'] else (excluded_record['year_month'] if data_stats.get('is_year_month_converted') else ex_d_str)
        p_label = t("col_cdc_yw", lang) if data_stats['is_year_week_converted'] else (t("col_cdc_ym", lang) if data_stats.get('is_year_month_converted') else t("axis_date", lang))
        ex_hover = t("hover_excluded_incomplete", lang, period_label=p_label, period_val=ex_w, val=ex_v)

        fig.add_trace(go.Scatter(
            x=[ex_d_str],
            y=transform_y([ex_v]),
            customdata=[ex_w],
            mode='markers',
            name=t("trace_excluded", lang),
            marker=dict(size=12, color='rgba(0,0,0,0)', symbol='circle', line=dict(width=2.5, color='#E53935')),
            hovertemplate=ex_hover
        ))

    # Ground Truth overlay if in backtesting mode
    if is_backtesting and ground_truth_df is not None:
        all_gt_dates = [hist_dates[-1]] + future_dates_list
        all_gt_vals = [hist_vals[-1]] + list(ground_truth_df['y'].values)
        all_gt_yws = [hist_yws[-1]] + future_yws_list
        if data_stats['is_year_week_converted']:
            cd_gt = np.stack([all_gt_yws, all_gt_vals], axis=-1)
        else:
            cd_gt = np.stack([all_gt_dates, all_gt_vals], axis=-1)
        fig.add_trace(go.Scatter(
            x=all_gt_dates,
            y=transform_y(all_gt_vals),
            customdata=cd_gt,
            mode='lines+markers',
            name=t("trace_ground_truth", lang),
            line=dict(color='#E53935', width=3.5, dash='dot'),
            marker=dict(size=8, color='#E53935', symbol='star'),
            hovertemplate=gt_hover
        ))

    # Vertical separation line
    fig.add_vline(
        x=hist_dates[-1],
        line_width=1.5,
        line_dash="dash",
        line_color="#78909C",
        annotation_text=t("forecast_start_line", lang),
        annotation_position="top left",
        annotation_font=dict(size=11, color="#90A4AE")
    )

    x_axis_title = t("axis_ym", lang) if data_stats.get('is_year_month_converted') else (t("axis_yw", lang) if data_stats['is_year_week_converted'] else t("axis_date", lang))

    fig.update_layout(
        xaxis=dict(
            title=dict(text=x_axis_title, standoff=12),
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label=t("btn_6m", lang), step="month", stepmode="backward"),
                    dict(count=1, label=t("btn_1y", lang), step="year", stepmode="backward"),
                    dict(count=2, label=t("btn_2y", lang), step="year", stepmode="backward"),
                    dict(step="all", label=t("btn_all", lang))
                ]),
                y=1.05,
                x=0.0,
                font=dict(size=11),
                bgcolor="rgba(128, 128, 128, 0.15)",
                activecolor="rgba(30, 136, 229, 0.35)"
            ),
            rangeslider=dict(visible=False),
            type="date"
        ),
        yaxis=dict(
            title=dict(text=f"{target_disp} {t('scale_log_suffix', lang)}" if is_log_scale else str(target_disp)),
            autorange=True
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        height=500,
        margin=dict(l=35, r=20, t=35, b=75)
    )

    if is_log_scale:
        fig.update_yaxes(
            type="log",
            tickvals=[0.2, 1, 10, 100, 1000, 3000],
            ticktext=["0", "1", "10", "100", "1,000", "3,000"]
        )

    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": True})

    # Adaptive decimal formatting based on value range (e.g. 3 decimals for exchange rates, 1 for cases)
    val_max = float(np.max(clean_df['y'])) if len(clean_df) > 0 else 100.0
    dec = 3 if val_max < 100 else 1
    dec_rng = 3 if val_max < 100 else 0

    # 3 Summary Cards below chart
    c1, c2, c3 = st.columns(3)
    card_colon = t("colon", lang)
    with c1:
        st.markdown(f"""
        <div class="metric-card-tfm">
            <h4>🔵 {t("card_tfm_title", lang)}</h4>
            <p><b>{t("card_mean", lang, horizon=horizon)}{card_colon}</b>{np.mean(res_tfm['forecast']):,.{dec}f}</p>
            <p><b>{t("card_range", lang)}{card_colon}</b>{np.min(res_tfm['forecast']):,.{dec_rng}f} ~ {np.max(res_tfm['forecast']):,.{dec_rng}f}</p>
            <p><b>{t("card_latency", lang)}{card_colon}</b>{res_tfm['elapsed_time_sec']} {t("unit_seconds", lang)} ({html.escape(res_tfm['device_used'].upper())})</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card-prophet">
            <h4>🟠 {t("card_prophet_title", lang)}</h4>
            <p><b>{t("card_mean", lang, horizon=horizon)}{card_colon}</b>{np.mean(res_prophet['forecast']):,.{dec}f}</p>
            <p><b>{t("card_range", lang)}{card_colon}</b>{np.min(res_prophet['forecast']):,.{dec_rng}f} ~ {np.max(res_prophet['forecast']):,.{dec_rng}f}</p>
            <p><b>{t("card_latency", lang)}{card_colon}</b>{res_prophet['elapsed_time_sec']} {t("unit_seconds", lang)}</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card-arima">
            <h4>🟢 {t("card_arima_title", lang)}</h4>
            <p><b>{t("card_mean", lang, horizon=horizon)}{card_colon}</b>{np.mean(res_arima['forecast']):,.{dec}f}</p>
            <p><b>{t("card_order_aic", lang)}{card_colon}</b>{html.escape(str(res_arima['order_str']))} (AIC: {res_arima['aic']})</p>
            <p><b>{t("card_latency", lang)}{card_colon}</b>{res_arima['elapsed_time_sec']} {t("unit_seconds", lang)}</p>
        </div>
        """, unsafe_allow_html=True)

    # 歷年同期季節性流行趨勢疊加分析 (Seasonality Multi-Year Overlay, for Weekly & Monthly series)
    min_required_len = 24 if (selected_freq == 'M' or data_stats.get('is_year_month_converted')) else 52
    show_overlay = (
        (selected_freq in ['W', 'M'] or data_stats.get('is_year_week_converted') or data_stats.get('is_year_month_converted'))
        and 'full_history_df' in locals()
        and len(full_history_df) >= min_required_len
    )
    if show_overlay:
        is_monthly_overlay = (selected_freq == 'M') or data_stats.get('is_year_month_converted', False)
        unit_label = t("unit_month", lang) if is_monthly_overlay else t("unit_week", lang)
        range_label = t("range_months", lang) if is_monthly_overlay else t("range_weeks", lang)

        with st.expander(t("overlay_expander_title", lang, unit=unit_label), expanded=False):
            st.markdown(t("overlay_guidance", lang, range=range_label))

            # Create multi-year dataframe
            overlay_df = full_history_df.copy()
            if is_monthly_overlay:
                overlay_df['cal_year'] = overlay_df['ds'].dt.year
                overlay_df['cal_period'] = overlay_df['ds'].dt.month
            elif 'year_week' in overlay_df and overlay_df['year_week'].iloc[0] is not None and not data_stats.get('is_year_month_converted'):
                overlay_df['cal_year'] = overlay_df['year_week'].astype(str).str[:4].astype(int)
                overlay_df['cal_period'] = overlay_df['year_week'].astype(str).str[4:].astype(int)
            else:
                overlay_df['cal_year'] = overlay_df['ds'].dt.year
                overlay_df['cal_period'] = overlay_df['ds'].dt.isocalendar().week.astype(int)

            fig_multi = go.Figure()
            palette = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1', '#5D4037', '#00897B', '#F4511E', '#3949AB']
            unique_years = sorted(overlay_df['cal_year'].unique())

            c_m1, c_m2 = st.columns([0.7, 0.3])
            with c_m2:
                overlay_log = st.checkbox(t("overlay_log_chk", lang), value=False, key="chk_overlay_log")

            for idx, yr in enumerate(unique_years):
                yr_data = overlay_df[overlay_df['cal_year'] == yr].sort_values('cal_period')
                yr_color = palette[idx % len(palette)]
                is_latest_year = (yr == max(unique_years))
                y_plot = [max(float(val), 0.2) for val in yr_data['y']] if overlay_log else yr_data['y']

                fig_multi.add_trace(go.Scatter(
                    x=yr_data['cal_period'],
                    y=y_plot,
                    customdata=yr_data['y'],
                    mode='lines+markers',
                    name=t("overlay_trace_name", lang, yr=yr, peak=f"{yr_data['y'].max():,.0f}", sum=f"{yr_data['y'].sum():,.0f}", cases=t("unit_cases", lang)),
                    line=dict(color=yr_color, width=3.5 if is_latest_year else 2.0),
                    marker=dict(size=6 if is_latest_year else 4, color=yr_color),
                    hovertemplate=t("overlay_hover", lang, yr=yr, unit=unit_label, cases=t("unit_cases", lang))
                ))

            x_title = f"{unit_label} ({range_label})"
            x_dtick = 1 if is_monthly_overlay else 5
            x_range = [1, 12] if is_monthly_overlay else [1, 53]
            fig_multi.update_layout(
                title=dict(text=f"<b>{t('overlay_chart_title', lang, unit=unit_label, range=range_label)}</b>", x=0.02, y=0.98),
                xaxis=dict(title=dict(text=x_title, standoff=10), tickmode='linear', dtick=x_dtick, range=x_range),
                yaxis=dict(title=f"{t('unit_cases', lang)} {t('scale_log_suffix', lang) if overlay_log else t('scale_linear_suffix', lang)}"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center", font=dict(size=11)),
                height=460,
                margin=dict(l=35, r=20, t=45, b=75)
            )
            if overlay_log:
                fig_multi.update_yaxes(
                    type="log",
                    tickvals=[0.2, 1, 10, 100, 1000, 3000],
                    ticktext=["0", "1", "10", "100", "1,000", "3,000"]
                )
            st.plotly_chart(fig_multi, use_container_width=True, config={"responsive": True, "displayModeBar": True})

            # Annual comparison metrics table
            summary_records = []
            for yr in unique_years:
                sub = overlay_df[overlay_df['cal_year'] == yr]
                peak_p = sub.loc[sub['y'].idxmax()]['cal_period'] if len(sub) > 0 else 0
                summary_records.append({
                    t("th_year", lang): f"{yr} {t('unit_year', lang)}" if lang == 'zh' else str(yr),
                    t("th_stat_periods", lang, unit=unit_label): f"{len(sub)} {unit_label}",
                    t("th_annual_sum", lang): f"{sub['y'].sum():,.0f} {t('unit_cases', lang)}",
                    t("th_peak_val", lang, unit=unit_label): f"{sub['y'].max():,.0f} {t('unit_cases', lang)}",
                    t("th_peak_timing", lang, unit=unit_label): f"{t('prefix_period', lang)}{peak_p} {unit_label}",
                    t("th_mean_val", lang, unit=unit_label): f"{sub['y'].mean():,.1f} {t('unit_cases', lang)}/{unit_label}"
                })
            st.dataframe(pd.DataFrame(summary_records), use_container_width=True, hide_index=True)


with tab2:
    st.markdown(f"### 🏆 {t('tab2_header', lang)}")

    if is_backtesting:
        st.success(t("backtest_success_msg", lang, horizon=horizon))

        metrics_comparison = []

        metric_names = [
            (t("metric_mae", lang), "MAE", t("pref_lower", lang), t("unit_cases", lang)),
            (t("metric_rmse", lang), "RMSE", t("pref_lower", lang), t("unit_cases", lang)),
            (t("metric_mape", lang), "MAPE (%)", t("pref_lower", lang), "%"),
            (t("metric_smape", lang), "SMAPE (%)", t("pref_lower", lang), "%"),
            (t("metric_wape", lang), "WAPE (%)", t("pref_lower", lang), "%"),
            (t("metric_dir_acc", lang), "Directional Accuracy (%)", t("pref_higher", lang), "%"),
            (t("metric_peak_err", lang), "Peak Value Error (%)", t("pref_lower", lang), "%"),
            (t("metric_timing_err", lang), "Peak Timing Error (Steps)", t("pref_lower", lang), f" {unit_label}"),
            (t("metric_ci_cov", lang), "CI Coverage (%)", t("pref_higher", lang), "%")
        ]

        pref_lower_str = t("pref_lower", lang)

        for display_name, key, pref, unit in metric_names:
            v_tfm = metrics_tfm.get(key, None)
            v_pro = metrics_prophet.get(key, None)
            v_ari = metrics_arima.get(key, None)

            if v_tfm is None or v_pro is None or v_ari is None:
                continue

            vals = {"TimesFM 3.0": v_tfm, "Prophet": v_pro, "Auto ARIMA": v_ari}
            if pref == pref_lower_str:
                best_model = min(vals, key=vals.get)
            else:
                best_model = max(vals, key=vals.get)

            winner_str = f"🏆 {best_model}"

            metrics_comparison.append({
                t("th_eval_metric", lang): display_name,
                "Google TimesFM 3.0": f"{v_tfm:,.2f} {unit}" if isinstance(v_tfm, float) else f"{v_tfm} {unit}",
                "Meta Prophet": f"{v_pro:,.2f} {unit}" if isinstance(v_pro, float) else f"{v_pro} {unit}",
                "Auto ARIMA": f"{v_ari:,.2f} {unit}" if isinstance(v_ari, float) else f"{v_ari} {unit}",
                t("th_winner", lang): winner_str,
                t("th_criteria", lang): pref
            })

        df_metrics = pd.DataFrame(metrics_comparison)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # Responsive Bar Chart for Mobile & Desktop (clean single chart with selector)
        st.markdown(f"#### 📊 {t('bar_chart_header', lang)}")
        opt_mae = t("bar_mae", lang)
        opt_mape = t("bar_mape", lang)
        opt_dir = t("bar_dir_acc", lang)
        chart_view = st.radio(
            t("bar_toggle_label", lang),
            [opt_mae, opt_mape, opt_dir],
            horizontal=True
        )

        fig_metric_bar = go.Figure()

        if chart_view == opt_mae:
            metric_vals = [metrics_tfm['MAE'], metrics_prophet['MAE'], metrics_arima['MAE']]
            title_text = t("bar_title_mae", lang)
        elif chart_view == opt_mape:
            metric_vals = [metrics_tfm['MAPE (%)'], metrics_prophet['MAPE (%)'], metrics_arima['MAPE (%)']]
            title_text = t("bar_title_mape", lang)
        else:
            metric_vals = [metrics_tfm['Directional Accuracy (%)'], metrics_prophet['Directional Accuracy (%)'], metrics_arima['Directional Accuracy (%)']]
            title_text = t("bar_title_dir_acc", lang)

        fig_metric_bar.add_trace(go.Bar(
            x=['TimesFM 3.0', 'Prophet', 'Auto ARIMA'],
            y=metric_vals,
            marker_color=['#1E88E5', '#FF9800', '#2E7D32'],
            text=[f"{v:,.1f}" for v in metric_vals],
            textposition='outside'
        ))

        fig_metric_bar.update_layout(
            title=f"<b>{title_text}</b>",
            height=360,
            margin=dict(l=20, r=20, t=50, b=30),
            yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
        )
        st.plotly_chart(fig_metric_bar, use_container_width=True, config={"responsive": True, "displayModeBar": False})

    else:
        st.info(t("future_mode_info", lang))

        val_max = float(np.max(clean_df['y'])) if len(clean_df) > 0 else 100.0
        dec = 3 if val_max < 100 else 1

        stat_compare = pd.DataFrame({
            t("th_stat_feature", lang): [
                t("stat_horizon", lang),
                t("stat_peak", lang),
                t("stat_min", lang),
                t("stat_mean", lang),
                t("stat_sum", lang),
                t("stat_latency", lang)
            ],
            "Google TimesFM 3.0": [
                f"{horizon} {unit_label}",
                f"{np.max(res_tfm['forecast']):,.{dec}f}",
                f"{np.min(res_tfm['forecast']):,.{dec}f}",
                f"{np.mean(res_tfm['forecast']):,.{dec}f}",
                f"{np.sum(res_tfm['forecast']):,.{dec}f}",
                f"{res_tfm['elapsed_time_sec']}s"
            ],
            "Meta Prophet": [
                f"{horizon} {unit_label}",
                f"{np.max(res_prophet['forecast']):,.{dec}f}",
                f"{np.min(res_prophet['forecast']):,.{dec}f}",
                f"{np.mean(res_prophet['forecast']):,.{dec}f}",
                f"{np.sum(res_prophet['forecast']):,.{dec}f}",
                f"{res_prophet['elapsed_time_sec']}s"
            ],
            "Auto ARIMA": [
                f"{horizon} {unit_label}",
                f"{np.max(res_arima['forecast']):,.{dec}f}",
                f"{np.min(res_arima['forecast']):,.{dec}f}",
                f"{np.mean(res_arima['forecast']):,.{dec}f}",
                f"{np.sum(res_arima['forecast']):,.{dec}f}",
                f"{res_arima['elapsed_time_sec']}s"
            ]
        })
        st.dataframe(stat_compare, use_container_width=True, hide_index=True)


with tab3:
    st.markdown(f"### 🧠 {t('tab3_header', lang)}")

    last_val = last_hist_val
    tfm_end = res_tfm['forecast'][-1]
    pro_end = res_prophet['forecast'][-1]
    ari_end = res_arima['forecast'][-1]

    tfm_change_pct = ((tfm_end - last_val) / (last_val + 1e-5)) * 100
    pro_change_pct = ((pro_end - last_val) / (last_val + 1e-5)) * 100
    ari_change_pct = ((ari_end - last_val) / (last_val + 1e-5)) * 100

    def get_stage_desc(pct, cur_lang):
        if pct > 25:
            return t("stage_strong_up", cur_lang)
        elif pct > 5:
            return t("stage_mild_up", cur_lang)
        elif abs(pct) <= 5:
            return t("stage_plateau", cur_lang)
        elif pct > -25:
            return t("stage_mild_down", cur_lang)
        else:
            return t("stage_rapid_down", cur_lang)

    baseline_label = (
        f"{t('prefix_ym', lang)} {base_working_df['year_month'].iloc[-1]}"
        if data_stats.get('is_year_month_converted')
        else (
            f"{t('prefix_yw', lang)} {base_working_df['year_week'].iloc[-1]}"
            if data_stats['is_year_week_converted']
            else last_hist_date.strftime('%Y-%m-%d')
        )
    )
    val_max = float(np.max(clean_df['y'])) if len(clean_df) > 0 else 100.0
    dec = 3 if val_max < 100 else 1
    dec_pct = 2 if val_max < 100 else 1

    sec1_title = t("report_sec1_title", lang)
    baseline_title = t("report_baseline_title", lang)
    assess_title = t("report_assessment", lang)
    pred_step_str = t("report_pred_step", lang, horizon=horizon)
    chg_rate_str = t("report_change_rate", lang)
    sec2_title = t("report_sec2_title", lang)
    p1_title = t("report_p1_title", lang)
    phenom_label = t("report_phenom", lang)
    p1_phenom = t("report_p1_phenom", lang)
    protect_label = t("report_protect", lang)
    p1_protect = t("report_p1_protect", lang)
    p2_title = t("report_p2_title", lang)
    p2_desc = t("report_p2_desc", lang)
    p3_title = t("report_p3_title", lang)
    p3_desc = t("report_p3_desc", lang)

    rep_colon = t("colon", lang)
    l_paren = "（" if lang == 'zh' else " ("
    r_paren = "）" if lang == 'zh' else ")"

    st.markdown(f"""
    #### 1. {sec1_title}
    - **{baseline_title}{rep_colon}** `{last_val:,.{dec}f}`{l_paren}{baseline_label}{r_paren}
    - **Google TimesFM 3.0 {assess_title}{rep_colon}** {pred_step_str} `{tfm_end:,.{dec}f}`{l_paren}{chg_rate_str} `{tfm_change_pct:+.{dec_pct}f}%`{r_paren} ➜ {get_stage_desc(tfm_change_pct, lang)}
    - **Meta Prophet {assess_title}{rep_colon}** {pred_step_str} `{pro_end:,.{dec}f}`{l_paren}{chg_rate_str} `{pro_change_pct:+.{dec_pct}f}%`{r_paren} ➜ {get_stage_desc(pro_change_pct, lang)}
    - **Auto ARIMA {assess_title}{rep_colon}** {pred_step_str} `{ari_end:,.{dec}f}`{l_paren}{chg_rate_str} `{ari_change_pct:+.{dec_pct}f}%`{r_paren} ➜ {get_stage_desc(ari_change_pct, lang)}

    ---

    #### 2. {sec2_title}
    1. **{p1_title}**{rep_colon}
       - **{phenom_label}{rep_colon}** {p1_phenom}
       - **{protect_label}{rep_colon}** {p1_protect}
    2. **{p2_title}**{rep_colon}
       - {p2_desc}
    3. **{p3_title}**{rep_colon}
       - {p3_desc}
    """)


with tab4:
    st.markdown(f"### 📋 {t('tab4_header', lang, horizon=horizon)}")

    date_col_title = (
        t("col_pred_month_start", lang)
        if data_stats.get('is_year_month_converted')
        else (
            t("col_pred_week_start", lang)
            if data_stats['is_year_week_converted']
            else t("col_pred_date", lang)
        )
    )
    export_dict = {
        t("col_step", lang): [f"+{i+1}" for i in range(horizon)],
        date_col_title: future_dates_list,
    }

    if data_stats.get('is_year_month_converted'):
        export_dict[t("col_cdc_ym", lang)] = future_yws_list
    elif data_stats['is_year_week_converted']:
        export_dict[t("col_cdc_yw", lang)] = future_yws_list

    val_max = float(np.max(clean_df['y'])) if len(clean_df) > 0 else 100.0
    dec = 3 if val_max < 100 else 1

    export_dict.update({
        t("col_tfm_pred", lang): np.round(res_tfm['forecast'], dec),
        t("col_tfm_ci_low", lang): np.round(res_tfm['ci_lower'], dec),
        t("col_tfm_ci_high", lang): np.round(res_tfm['ci_upper'], dec),
        t("col_pro_pred", lang): np.round(res_prophet['forecast'], dec),
        t("col_pro_ci_low", lang): np.round(res_prophet['ci_lower'], dec),
        t("col_pro_ci_high", lang): np.round(res_prophet['ci_upper'], dec),
        t("col_ari_pred", lang): np.round(res_arima['forecast'], dec),
        t("col_ari_ci_low", lang): np.round(res_arima['ci_lower'], dec),
        t("col_ari_ci_high", lang): np.round(res_arima['ci_upper'], dec),
    })

    if is_backtesting and ground_truth_df is not None:
        gt_vals = ground_truth_df['y'].values
        export_dict[t("col_actual", lang)] = np.round(gt_vals, dec)
        export_dict[t("col_tfm_err", lang)] = np.round(np.abs(res_tfm['forecast'] - gt_vals), dec)
        export_dict[t("col_pro_err", lang)] = np.round(np.abs(res_prophet['forecast'] - gt_vals), dec)
        export_dict[t("col_ari_err", lang)] = np.round(np.abs(res_arima['forecast'] - gt_vals), dec)

    df_export = pd.DataFrame(export_dict)
    st.dataframe(df_export, use_container_width=True, hide_index=True)

    # Sanitize dataframe against CSV Formula / DDE Injection before export
    df_export_safe = sanitize_dataframe_for_csv_export(df_export)

    csv_buffer = io.StringIO()
    df_export_safe.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

    st.download_button(
        label=t("btn_download_csv", lang, horizon=horizon),
        data=csv_bytes,
        file_name=f"epidemic_forecast_{horizon}steps_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help=t("btn_download_help", lang)
    )
