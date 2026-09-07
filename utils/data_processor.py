"""
Data Processor Module for Epidemic Time Series.
Features:
1. Exact Taiwan CDC (疾管署) & MMWR EpiWeek mathematical rule engine:
   - Weeks run Sunday to Saturday.
   - The week containing Wednesday determines the Epidemiological Year (Week 1 contains Wednesday).
   - Generates and converts any date <-> year-week for ANY past or future year (infinite year coverage without file expiration).
2. Automated detection of Taiwan CDC '發病年週' (e.g. '202434', '202501').
3. Strict security controls:
   - Path traversal prevention on local file loading.
   - Row limit and size enforcement to prevent Denial of Service (DoS).
   - CSV formula injection (DDE) sanitization for secure report exports.
4. Clean preprocessing, frequency inference, missing value interpolation, and epidemiology statistics.
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import io
import os
import re

MAX_ALLOWABLE_ROWS = 100000
MAX_ALLOWABLE_BYTES = 50 * 1024 * 1024 # 50 MB
ALLOWED_BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


# ---------------- SECURITY SANITIZATION UTILITIES ----------------

def validate_safe_path(filepath: str) -> str:
    """
    Validates that a requested file path resolves safely within the project workspace.
    Prevents directory traversal (Path Traversal / Local File Inclusion).
    """
    real_path = os.path.realpath(filepath)
    # Check if the resolved path starts with the allowed workspace base directory
    try:
        common = os.path.commonpath([real_path, ALLOWED_BASE_DIR])
        if common != ALLOWED_BASE_DIR:
            raise PermissionError(f"Security Alert: Path traversal attempt detected outside workspace: '{filepath}'")
    except ValueError:
        raise PermissionError(f"Security Alert: Invalid path traversal detected: '{filepath}'")

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check file size
    if os.path.getsize(real_path) > MAX_ALLOWABLE_BYTES:
        raise ValueError(f"File size exceeds allowable maximum of {MAX_ALLOWABLE_BYTES / (1024*1024):.0f}MB.")

    return real_path


def sanitize_dataframe_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes string cells in a dataframe before exporting to CSV.
    Prevents CSV Formula Injection / Dynamic Data Exchange (DDE) attacks
    in Microsoft Excel, LibreOffice Calc, or Google Sheets (OWASP CSV Injection).
    """
    clean_df = df.copy()
    dangerous_prefixes = ('=', '+', '-', '@', '\t', '\r')

    for col in clean_df.columns:
        if clean_df[col].dtype == object or pd.api.types.is_string_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].apply(
                lambda val: f"'{val}" if isinstance(val, str) and val.startswith(dangerous_prefixes) else val
            )
    return clean_df


# ---------------- TAIWAN CDC EPIWEEK MATHEMATICAL ENGINE ----------------

def get_cdc_week1_sunday(year: int) -> date:
    """
    Returns the Sunday date of Week 1 for a given epidemiological year under Taiwan CDC / MMWR rules:
    - An epidemiological week begins on Sunday and ends on Saturday.
    - Week 1 is the week that contains the first Wednesday of that year (or Wednesday of that week is in `year`).
    """
    jan1 = date(year, 1, 1)
    days_since_sun = (jan1.weekday() + 1) % 7
    sun = jan1 - timedelta(days=days_since_sun)
    wed = sun + timedelta(days=3)
    if wed.year == year:
        return sun
    else:
        return sun + timedelta(days=7)


def cdc_year_week_to_date(year_week_val: Union[str, int]) -> str:
    """
    Converts Taiwan CDC Year-Week (e.g. '202434', '202501', 202434) to Week Start Date (Sunday, 'YYYY-MM-DD').
    Works for ANY year (no expiration across new year transitions).
    """
    raw_str = str(year_week_val).strip()
    clean_str = raw_str.replace('-', '').replace('_', '').replace('W', '')

    if clean_str.isdigit():
        if len(clean_str) == 5:
            clean_str = f"{clean_str[:4]}0{clean_str[4:]}"
        elif len(clean_str) > 6:
            clean_str = clean_str[:6]

    if len(clean_str) == 6 and clean_str.isdigit():
        try:
            year = int(clean_str[:4])
            week = int(clean_str[4:])
            w1_sun = get_cdc_week1_sunday(year)
            target_sunday = w1_sun + timedelta(days=(week - 1) * 7)
            return target_sunday.strftime('%Y-%m-%d')
        except (ValueError, OverflowError):
            return raw_str

    return raw_str


def date_to_cdc_year_week(dt_val: Union[datetime.datetime, datetime.date, pd.Timestamp, str]) -> str:
    """
    Converts any datetime/date to Taiwan CDC Year-Week string (e.g. '202636').
    Works for ANY year with 100% mathematical fidelity to Taiwan CDC's DIM_CAL standard.
    """
    if isinstance(dt_val, str):
        try:
            dt = pd.to_datetime(dt_val).date()
        except (ValueError, TypeError, pd.errors.ParserError):
            return str(dt_val)
    elif isinstance(dt_val, pd.Timestamp):
        dt = dt_val.date()
    elif isinstance(dt_val, datetime.datetime):
        dt = dt_val.date()
    else:
        dt = dt_val

    # Find the Sunday of the current week
    days_since_sun = (dt.weekday() + 1) % 7
    sunday = dt - timedelta(days=days_since_sun)
    # Wednesday determines the epidemiological year
    wednesday = sunday + timedelta(days=3)
    epi_year = wednesday.year

    w1_sun = get_cdc_week1_sunday(epi_year)
    week_num = ((sunday - w1_sun).days // 7) + 1
    return f"{epi_year}{week_num:02d}"


def generate_dim_cal_dataframe(start_year: int = 2007, end_year: int = 2035) -> pd.DataFrame:
    """
    Generates a full Taiwan CDC calendar mapping DataFrame matching DIM_CAL.csv format.
    Columns: CAL_YMD (int), CAL_YEAR (int), CAL_WEEK (int), Year_Week (int).
    Can be used to generate calendar files for any arbitrary future year (e.g. 2026-2050).
    """
    records = []
    curr = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)

    while curr <= end_date:
        cal_ymd = int(curr.strftime('%Y%m%d'))
        days_since_sun = (curr.weekday() + 1) % 7
        sunday = curr - timedelta(days=days_since_sun)
        wednesday = sunday + timedelta(days=3)
        epi_year = wednesday.year
        w1_sun = get_cdc_week1_sunday(epi_year)
        week_num = ((sunday - w1_sun).days // 7) + 1
        yw_int = epi_year * 100 + week_num

        records.append({
            'CAL_YMD': cal_ymd,
            'CAL_YEAR': epi_year,
            'CAL_WEEK': week_num,
            'Year_Week': yw_int
        })
        curr += timedelta(days=1)

    return pd.DataFrame(records)


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
            try:
                year = int(v_str[:4])
                week = int(v_str[4:])
                if 1990 <= year <= 2099 and 1 <= week <= 53:
                    match_count += 1
            except ValueError:
                continue

    return (match_count / len(non_null)) >= 0.7


# ---------------- DATASET LOADING & DETECTION ----------------

def load_dataset(file_obj_or_path) -> pd.DataFrame:
    """
    Safely loads dataset from file object, path, or bytes (CSV/Excel/JSON).
    Includes path traversal validation and size boundaries.
    """
    df = None
    if isinstance(file_obj_or_path, str):
        safe_path = validate_safe_path(file_obj_or_path)
        if safe_path.endswith('.csv'):
            df = pd.read_csv(safe_path)
        elif safe_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(safe_path)
        elif safe_path.endswith('.json'):
            df = pd.read_json(safe_path)
        else:
            df = pd.read_csv(safe_path)
    elif hasattr(file_obj_or_path, 'name'):
        # Streamlit UploadedFile object
        if hasattr(file_obj_or_path, 'size') and file_obj_or_path.size > MAX_ALLOWABLE_BYTES:
            raise ValueError(f"Uploaded file exceeds {MAX_ALLOWABLE_BYTES / (1024*1024):.0f}MB limit.")
        name = file_obj_or_path.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(file_obj_or_path)
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_obj_or_path)
        elif name.endswith('.json'):
            df = pd.read_json(file_obj_or_path)
        else:
            df = pd.read_csv(file_obj_or_path)
    elif isinstance(file_obj_or_path, (bytes, bytearray)):
        if len(file_obj_or_path) > MAX_ALLOWABLE_BYTES:
            raise ValueError("Input data stream exceeds allowable size limit.")
        df = pd.read_csv(io.BytesIO(file_obj_or_path))
    elif isinstance(file_obj_or_path, pd.DataFrame):
        df = file_obj_or_path.copy()
    else:
        raise ValueError("Unsupported data source format.")

    if len(df) > MAX_ALLOWABLE_ROWS:
        raise ValueError(f"Dataset contains {len(df):,} rows, exceeding maximum limit of {MAX_ALLOWABLE_ROWS:,} rows.")

    return df


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Intelligently identifies the most likely Date (or Year-Week) and Target Value columns.
    Returns (detected_date_col, detected_target_col, all_columns).
    """
    all_cols = list(df.columns)
    date_candidates = [
        '發病年週', '年週', 'year_week', 'yearweek', 'epiweek', 'week',
        'date', 'time', 'datetime', 'day', 'month', 'year',
        '日期', '時間', '週次', '月份', '監測日期', '通報日期'
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
            except (ValueError, TypeError, pd.errors.ParserError):
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
            except (ValueError, TypeError):
                continue

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
                except (ValueError, TypeError):
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
      using exact mathematical EpiWeek rules (works indefinitely across new years).
    - Converts target values to non-negative floats and interpolates missing values.
    - Calculates epidemiology statistics.
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
        except (ValueError, TypeError, pd.errors.ParserError):
            clean_df['ds'] = pd.to_datetime(raw_date_series, errors='coerce')
        converted_year_week = False

    clean_df['y'] = pd.to_numeric(df[target_col], errors='coerce')

    # Drop rows with invalid dates and sort chronologically
    clean_df = clean_df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

    # Populate reverse Year-Week for non-year-week series
    if not converted_year_week:
        clean_df['year_week'] = clean_df['ds'].apply(date_to_cdc_year_week)

    # Handle missing target values with linear interpolation
    clean_df['y'] = clean_df['y'].interpolate(method='linear').bfill().ffill()
    clean_df['y'] = np.maximum(clean_df['y'].values, 0.0)

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


if __name__ == '__main__':
    print("Generating Taiwan CDC DIM_CAL calendar table (2007-2050)...")
    df_new_cal = generate_dim_cal_dataframe(2007, 2050)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'DIM_CAL.csv')
    df_new_cal.to_csv(out_path, index=False)
    print(f"Successfully generated and saved {len(df_new_cal)} rows to {out_path}!")
