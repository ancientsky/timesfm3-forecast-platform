"""
Data Processor Module for Epidemic Time Series.
Supports Taiwan CDC (疾管署) Year-Week (年週, 如 202434) automated detection
and conversion to week start dates using DIM_CAL calendar mapping,
as well as standard date parsing, missing value interpolation, and statistics.
"""

from typing import Tuple, Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import io
import os

# Cache for DIM_CAL lookup table
_DIM_CAL_MAP: Optional[Dict[str, str]] = None
_DATE_TO_YW_MAP: Optional[Dict[str, str]] = None


def _load_dim_cal():
    """Loads DIM_CAL.csv and builds bidirectional Year-Week <-> Week Start Date mappings."""
    global _DIM_CAL_MAP, _DATE_TO_YW_MAP
    if _DIM_CAL_MAP is not None:
        return _DIM_CAL_MAP, _DATE_TO_YW_MAP

    _DIM_CAL_MAP = {}
    _DATE_TO_YW_MAP = {}
    cal_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'DIM_CAL.csv')

    if os.path.exists(cal_path):
        try:
            df_cal = pd.read_csv(cal_path, dtype=str)
            # Find the first date (week start date, Sunday) for each Year_Week
            # Group by Year_Week, take min of CAL_YMD
            df_grouped = df_cal.groupby('Year_Week')['CAL_YMD'].min().reset_index()
            for _, row in df_grouped.iterrows():
                yw = str(row['Year_Week']).strip()
                ymd = str(row['CAL_YMD']).strip()
                if len(ymd) == 8:
                    dt_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
                    _DIM_CAL_MAP[yw] = dt_str

            # Also map individual dates to Year_Week
            for _, row in df_cal.iterrows():
                ymd = str(row['CAL_YMD']).strip()
                yw = str(row['Year_Week']).strip()
                if len(ymd) == 8:
                    dt_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
                    _DATE_TO_YW_MAP[dt_str] = yw
        except Exception as e:
            print(f"[data_processor] Failed reading DIM_CAL.csv: {e}")

    return _DIM_CAL_MAP, _DATE_TO_YW_MAP


def cdc_year_week_to_date(year_week_val: Union[str, int]) -> str:
    """
    Converts Taiwan CDC Year-Week (e.g. '202434', '202501', 202434) to Week Start Date (Sunday, YYYY-MM-DD).
    Uses DIM_CAL calendar table with algorithmic MMWR fallback.
    """
    yw_map, _ = _load_dim_cal()
    raw_str = str(year_week_val).strip()
    clean_str = raw_str.replace('-', '').replace('_', '').replace('W', '')

    # Standardize to 6 digits (e.g. 202401)
    if clean_str.isdigit():
        if len(clean_str) == 5: # e.g. 20241 -> 202401
            clean_str = f"{clean_str[:4]}0{clean_str[4:]}"
        elif len(clean_str) > 6:
            clean_str = clean_str[:6]

    # 1. Lookup in DIM_CAL
    if clean_str in yw_map:
        return yw_map[clean_str]

    # 2. Algorithmic Fallback (CDC MMWR Week standard definition)
    try:
        if len(clean_str) == 6 and clean_str.isdigit():
            year = int(clean_str[:4])
            week = int(clean_str[4:])
            jan4 = datetime(year, 1, 4)
            w1_sun_offset = (jan4.weekday() + 1) % 7
            week1_sunday = jan4 - timedelta(days=w1_sun_offset)
            target_sunday = week1_sunday + timedelta(weeks=(week - 1))
            return target_sunday.strftime('%Y-%m-%d')
    except Exception:
        pass

    return raw_str


def date_to_cdc_year_week(dt: pd.Timestamp) -> str:
    """
    Converts a datetime (or Timestamp) to Taiwan CDC Year-Week string (e.g. '202636').
    """
    _, date_map = _load_dim_cal()
    dt_str = dt.strftime('%Y-%m-%d')

    if dt_str in date_map:
        return date_map[dt_str]

    # Algorithmic MMWR calculation
    sun_offset = (dt.weekday() + 1) % 7
    sunday = dt - timedelta(days=sun_offset)
    wednesday = sunday + timedelta(days=3)
    year = wednesday.year

    jan4 = datetime(year, 1, 4)
    w1_sun_offset = (jan4.weekday() + 1) % 7
    week1_sunday = jan4 - timedelta(days=w1_sun_offset)

    week = (sunday - week1_sunday).days // 7 + 1
    return f"{year}{week:02d}"


def is_year_week_series(series: pd.Series) -> bool:
    """
    Determines if a series contains Taiwan CDC 'Year-Week' numbers (e.g. 202434, 202501).
    """
    non_null = series.dropna().head(20)
    if len(non_null) == 0:
        return False

    match_count = 0
    for val in non_null:
        v_str = str(val).strip().replace('-', '').replace('_', '').replace('W', '')
        if v_str.isdigit() and len(v_str) in [5, 6]:
            year = int(v_str[:4])
            week = int(v_str[4:])
            if 1990 <= year <= 2099 and 1 <= week <= 53:
                match_count += 1

    return (match_count / len(non_null)) >= 0.7


def load_dataset(file_obj_or_path) -> pd.DataFrame:
    """Loads dataset from file object, path, or bytes (CSV/Excel/JSON)."""
    if isinstance(file_obj_or_path, str):
        if file_obj_or_path.endswith('.csv'):
            return pd.read_csv(file_obj_or_path)
        elif file_obj_or_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_obj_or_path)
        elif file_obj_or_path.endswith('.json'):
            return pd.read_json(file_obj_or_path)
        else:
            return pd.read_csv(file_obj_or_path)
    elif hasattr(file_obj_or_path, 'name'):
        name = file_obj_or_path.name.lower()
        if name.endswith('.csv'):
            return pd.read_csv(file_obj_or_path)
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_obj_or_path)
        elif name.endswith('.json'):
            return pd.read_json(file_obj_or_path)
        else:
            return pd.read_csv(file_obj_or_path)
    elif isinstance(file_obj_or_path, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_obj_or_path))
    elif isinstance(file_obj_or_path, pd.DataFrame):
        return file_obj_or_path.copy()
    else:
        raise ValueError("Unsupported data source format.")


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Intelligently identifies the most likely Date (or Year-Week) and Target Value columns.
    Returns (detected_date_col, detected_target_col, all_columns).
    """
    all_cols = list(df.columns)
    date_candidates = [
        '發病年週', '年週', 'year_week', 'yearweek', 'epiweek', 'week',
        'date', 'time', 'datetime', 'day', 'month', 'year',
        '日期', '時間', '週次', '年週', '月份', '監測日期', '通報日期'
    ]
    target_candidates = [
        '確定病例數', '確診病例數', 'case', 'cases', 'confirmed', 'count', 'value', 'target', 'y', 'rate',
        '確診', '病例', '就診', '人次', '個案', '陽性', '發生數', '數值'
    ]

    detected_date = None
    detected_target = None

    # Priority 1: Exact keyword match for Year-Week or Date
    for col in all_cols:
        col_lower = str(col).lower()
        if any(cand in col_lower for cand in date_candidates):
            detected_date = col
            break

    # Priority 2: Check if column values are Year-Week numbers or parseable datetimes
    if detected_date is None:
        for col in all_cols:
            if is_year_week_series(df[col]):
                detected_date = col
                break
            try:
                pd.to_datetime(df[col].dropna().head(5), format='mixed')
                detected_date = col
                break
            except Exception:
                continue

    # Find target numeric column
    for col in all_cols:
        if col == detected_date:
            continue
        col_lower = str(col).lower()
        if any(cand in col_lower for cand in target_candidates):
            try:
                pd.to_numeric(df[col].dropna().head(5))
                detected_target = col
                break
            except Exception:
                continue

    # If target not found by keyword, pick first numeric column
    if detected_target is None:
        for col in all_cols:
            if col == detected_date:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                detected_target = col
                break
            else:
                try:
                    pd.to_numeric(df[col].dropna().head(5))
                    detected_target = col
                    break
                except Exception:
                    continue

    if detected_date is None and len(all_cols) > 0:
        detected_date = all_cols[0]
    if detected_target is None and len(all_cols) > 1:
        detected_target = all_cols[1] if all_cols[1] != detected_date else all_cols[0]

    return detected_date, detected_target, all_cols


def infer_frequency(date_series: pd.Series) -> str:
    """
    Infers frequency: 'D' (Daily), 'W' (Weekly), 'M' (Monthly), or fallback 'D'.
    """
    if len(date_series) < 2:
        return 'D'

    dates = pd.to_datetime(date_series).sort_values()
    diffs = dates.diff().dropna().dt.total_seconds() / (24 * 3600)
    median_diff = diffs.median()

    if median_diff <= 1.5:
        return 'D'
    elif 1.5 < median_diff <= 10.0:
        return 'W'
    elif 10.0 < median_diff <= 45.0:
        return 'M'
    else:
        return 'W' if 5 <= median_diff <= 9 else 'D'


def prepare_epidemic_data(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    override_freq: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocesses epidemic dataframe:
    - Auto-detects and converts Taiwan CDC Year-Week ('發病年週', 如 202434) to Week Start Date (週日)
    - Converts target values to float and interpolates missing values
    - Calculates epidemiology statistics
    """
    clean_df = pd.DataFrame()
    raw_date_series = df[date_col].copy()

    # Check if this column is a Taiwan CDC Year-Week series
    is_yw = is_year_week_series(raw_date_series)

    if is_yw:
        clean_df['year_week'] = raw_date_series.astype(str).str.strip().str.replace('"', '')
        clean_df['ds'] = clean_df['year_week'].apply(cdc_year_week_to_date)
        clean_df['ds'] = pd.to_datetime(clean_df['ds'])
        converted_year_week = True
    else:
        clean_df['year_week'] = None
        try:
            clean_df['ds'] = pd.to_datetime(raw_date_series, format='mixed', errors='coerce')
        except Exception:
            clean_df['ds'] = pd.to_datetime(raw_date_series, errors='coerce')
        converted_year_week = False

    clean_df['y'] = pd.to_numeric(df[target_col], errors='coerce')

    # Drop rows with invalid dates and sort chronologically
    clean_df = clean_df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

    # If Year-Week column is present, also populate reverse for non-year-week series
    if not converted_year_week:
        clean_df['year_week'] = clean_df['ds'].apply(date_to_cdc_year_week)

    # Handle missing target values with linear interpolation
    clean_df['y'] = clean_df['y'].interpolate(method='linear').bfill().ffill()
    clean_df['y'] = np.maximum(clean_df['y'].values, 0.0) # Epidemic counts >= 0

    inferred_freq = ('W' if is_yw else infer_frequency(clean_df['ds'])) if override_freq is None else override_freq

    # Calculate epidemic descriptive stats
    y_vals = clean_df['y'].values
    peak_idx = int(np.argmax(y_vals))
    peak_date = clean_df['ds'].iloc[peak_idx]
    peak_val = float(y_vals[peak_idx])

    # Recent trend (last 4 steps vs previous 4 steps)
    if len(y_vals) >= 8:
        recent_mean = float(np.mean(y_vals[-4:]))
        prev_mean = float(np.mean(y_vals[-8:-4]))
        trend_pct = ((recent_mean - prev_mean) / (prev_mean + 1e-5)) * 100
    else:
        recent_mean = float(np.mean(y_vals))
        prev_mean = recent_mean
        trend_pct = 0.0

    stats = {
        'count': len(clean_df),
        'min': float(np.min(y_vals)),
        'max': float(np.max(y_vals)),
        'mean': float(np.mean(y_vals)),
        'std': float(np.std(y_vals)),
        'last_val': float(y_vals[-1]),
        'peak_date': peak_date,
        'peak_val': peak_val,
        'frequency': inferred_freq,
        'start_date': clean_df['ds'].iloc[0],
        'end_date': clean_df['ds'].iloc[-1],
        'recent_trend_pct': trend_pct,
        'is_year_week_converted': converted_year_week,
        'start_year_week': clean_df['year_week'].iloc[0],
        'end_year_week': clean_df['year_week'].iloc[-1]
    }

    return clean_df, stats


def generate_future_dates(last_date: pd.Timestamp, horizon: int = 8, freq: str = 'W') -> pd.DatetimeIndex:
    """Generates the next `horizon` date points starting after last_date."""
    if freq == 'D':
        offset = pd.Timedelta(days=1)
        start = last_date + offset
        return pd.date_range(start=start, periods=horizon, freq='D')
    elif freq == 'W':
        offset = pd.Timedelta(weeks=1)
        start = last_date + offset
        return pd.date_range(start=start, periods=horizon, freq='W-SUN')
    elif freq == 'M':
        start = last_date + pd.DateOffset(months=1)
        return pd.date_range(start=start, periods=horizon, freq='MS')
    else:
        offset = pd.Timedelta(weeks=1)
        start = last_date + offset
        return pd.date_range(start=start, periods=horizon, freq='W-SUN')


def future_dates_to_year_weeks(future_dates: List[Union[str, pd.Timestamp]]) -> List[str]:
    """Converts a list of future Timestamp or date strings to CDC Year-Week strings."""
    res = []
    for d in future_dates:
        ts = pd.to_datetime(d)
        res.append(date_to_cdc_year_week(ts))
    return res
