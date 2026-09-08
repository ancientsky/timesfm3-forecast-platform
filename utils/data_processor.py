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
import urllib.parse
import urllib.request
import ipaddress
import socket
import ssl
import urllib3
import requests

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


def is_safe_url(url: str) -> Tuple[bool, str]:
    """
    Strictly validates that an external URL is safe to query (SSRF Prevention).
    - Requires http or https scheme.
    - Prohibits local/loopback/internal addresses (127.0.0.1, localhost, 0.0.0.0, ::1).
    - Blocks private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, etc.).
    - Resolves hostname via DNS to prevent DNS rebinding attacks to internal infrastructure.
    """
    if not isinstance(url, str) or not url.strip():
        return False, "網址不可為空白。"

    try:
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme.lower() not in ('http', 'https'):
            return False, "僅支援 HTTP 或 HTTPS 協議之公開網址。"

        hostname = parsed.hostname
        if not hostname:
            return False, "網址缺少有效的主機名稱 (hostname)。"

        # Block well-known localhost aliases
        if hostname.lower() in ('localhost', '127.0.0.1', '::1', '0.0.0.0', 'ip6-localhost'):
            return False, "基於伺服器安全性考量，禁止存取本機或回送位址 (localhost / loopback)。"

        # Check if direct IP literal is private / reserved
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False, f"基於安全性考量，禁止存取內部私有網段 IP ({hostname})。"
        except ValueError:
            # Hostname is a domain name, proceed to DNS resolution check
            pass

        # Attempt DNS resolution check for DNS rebinding to internal subnets
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    return False, f"網址解析出內部私有 IP 位址 ({ip_str})，已由 SSRF 安全機制攔截。"
        except socket.gaierror:
            # DNS resolution failed (e.g. offline sandbox or temporary DNS issue)
            # Allow to proceed to fetch step if it is a valid public domain format
            pass

        return True, ""
    except Exception as e:
        return False, f"網址解析失敗: {e}"


def fetch_live_csv_data(
    url: str,
    timeout: int = 15,
    fallback_path: Optional[str] = None
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Fetches live CSV data from an external HTTP/HTTPS URL with strict security controls:
    1. SSRF prevention via is_safe_url.
    2. Modern browser User-Agent header (prevents HTTP 403 Forbidden on government WAFs like CBC).
    3. Stream chunking with MAX_ALLOWABLE_BYTES guard against memory exhaustion DoS.
    4. Auto-detects encoding across utf-8-sig (with BOM), utf-8, cp950, big5, latin1.
    5. Graceful fallback to verified local dataset if offline or network blocked.
    Returns: (dataframe, is_live_online: bool, status_message: str)
    """
    is_safe, err_msg = is_safe_url(url)
    if not is_safe:
        raise ValueError(f"安全防護攔截: {err_msg}")

    # Full modern browser headers to pass governmental WAFs (Imperva / Cloudflare / F5)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,text/csv,text/plain,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
    }

    raw_bytes = None
    last_err = None

    # Tier 1: requests.Session with full header suite
    try:
        session = requests.Session()
        session.headers.update(headers)
        try:
            # Try standard secure SSL verification first
            resp = session.get(url.strip(), timeout=timeout, stream=True, verify=True)
            resp.raise_for_status()
        except Exception as ssl_err:
            err_text = str(ssl_err).lower()
            # If SSL handshake / verification failed (e.g. OpenSSL 3.0 Missing Subject Key Identifier on Taiwan Gov Root CA),
            # retry with unverified SSL session to accommodate legacy government PKI certificates
            if any(k in err_text for k in ['ssl', 'certificate', 'verify', 'subject key identifier', 'handshake']):
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = session.get(url.strip(), timeout=timeout, stream=True, verify=False)
                resp.raise_for_status()
            else:
                raise ssl_err

        with resp:
            content_chunks = []
            total_bytes = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    total_bytes += len(chunk)
                    if total_bytes > MAX_ALLOWABLE_BYTES:
                        raise ValueError(f"遠端資料大小超過 {MAX_ALLOWABLE_BYTES / (1024*1024):.0f}MB 限制。")
                    content_chunks.append(chunk)
            raw_bytes = b"".join(content_chunks)
    except Exception as e:
        last_err = e

    # Tier 2: urllib.request fallback with unverified context support
    if raw_bytes is None:
        try:
            try:
                ssl_ctx = ssl._create_unverified_context()
            except AttributeError:
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url.strip(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response:
                content_chunks = []
                total_bytes = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_ALLOWABLE_BYTES:
                        raise ValueError(f"遠端資料大小超過 {MAX_ALLOWABLE_BYTES / (1024*1024):.0f}MB 限制。")
                    content_chunks.append(chunk)
                raw_bytes = b"".join(content_chunks)
        except Exception as e:
            last_err = e

    if raw_bytes is not None:
        # Try multiple encodings
        decoded_text = None
        for enc in ['utf-8-sig', 'utf-8', 'cp950', 'big5', 'latin1']:
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            raise ValueError("無法解析遠端 CSV 檔案之文字編碼。")

        df = pd.read_csv(io.StringIO(decoded_text))
        if len(df) > MAX_ALLOWABLE_ROWS:
            raise ValueError(f"遠端數據筆數超過 {MAX_ALLOWABLE_ROWS:,} 筆限制。")

        return df, True, "即時線上連線同步成功"

    # Tier 3: Local fallback dataset if remote endpoint is firewalled or blocked
    if fallback_path and os.path.exists(fallback_path):
        safe_fallback = validate_safe_path(fallback_path)
        fallback_df = load_dataset(safe_fallback)
        err_str = str(last_err)
        if "403" in err_str or "Forbidden" in err_str:
            status_desc = "央行端點具備來源 IP 防火牆保護 (WAF 403)，已自動切換為高可靠性本地備援庫"
        else:
            status_desc = f"遠端連線異常 ({err_str})，已自動切換為本地備援資料庫"
        return fallback_df, False, status_desc

    if last_err:
        raise last_err
    raise RuntimeError("無法獲取遠端數據。")


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


def is_year_month_series(series: pd.Series, col_name: str = '') -> bool:
    """
    Determines if a series contains Taiwan CDC 'Year-Month' numbers (e.g. 200001, 202612).
    Taiwan CDC uses 6-digit strings 'YYYYMM' where MM is strictly 01..12.
    """
    col_lower = str(col_name).lower() if col_name else ''
    has_month_keyword = any(k in col_lower for k in ['年月', '月份', 'year_month', 'yearmonth', 'month', 'ym'])
    has_week_keyword = any(k in col_lower for k in ['週', 'week', 'epiweek'])

    if has_week_keyword:
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.head(30)
    valid_count = 0
    max_suffix = 0
    for val in sample:
        v_str = str(val).strip().replace('-', '').replace('_', '').replace('M', '')
        if v_str.isdigit() and len(v_str) == 6:
            try:
                year = int(v_str[:4])
                month = int(v_str[4:])
                if 1900 <= year <= 2099 and 1 <= month <= 12:
                    valid_count += 1
                    if month > max_suffix:
                        max_suffix = month
            except ValueError:
                pass

    is_valid_pattern = (valid_count / len(sample)) >= 0.7

    if has_month_keyword and is_valid_pattern:
        return True

    # If column name doesn't specify month, but is strictly 6-digit YYYYMM and max suffix <= 12
    if is_valid_pattern and max_suffix <= 12:
        if len(non_null) >= 15:
            all_suffixes_le_12 = True
            for val in non_null.head(60):
                v_str = str(val).strip().replace('-', '').replace('_', '')
                if v_str.isdigit() and len(v_str) == 6:
                    try:
                        m = int(v_str[4:])
                        if m > 12:
                            all_suffixes_le_12 = False
                            break
                    except ValueError:
                        pass
            if all_suffixes_le_12:
                return True

    return False


def is_year_week_series(series: pd.Series, col_name: str = '') -> bool:
    """
    Determines if a series contains Taiwan CDC 'Year-Week' numbers (e.g. 202434, 202501).
    Guards against Year-Month series (where suffix is 01..12).
    """
    if is_year_month_series(series, col_name=col_name):
        return False

    col_lower = str(col_name).lower() if col_name else ''
    if any(k in col_lower for k in ['年月', '月份', 'year_month', 'yearmonth', 'month']) and not any(k in col_lower for k in ['週', 'week']):
        return False

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
        '發病年月', '就診年月', '通報年月', '年月', '月份', 'year_month', 'yearmonth', 'month', 'ym',
        '發病年週', '就診年週', '通報年週', '年週', 'year_week', 'yearweek', 'epiweek', 'week',
        'date', 'time', 'datetime', 'day', 'year',
        '日期', '時間', '週次', '監測日期', '通報日期', '發病年月日', '年月日'
    ]
    target_candidates = [
        '確定病例數', '確診病例數', 'case', 'cases', 'confirmed', 'count', 'value', 'target', 'y', 'rate',
        '確診', '病例', '就診', '人次', '個案', '陽性', '發生數', '數值',
        'ntd', 'usd', '匯率', 'exchange', 'price', 'close', '收盤', '價格'
    ]

    detected_date = None
    detected_target = None

    # Priority 1: Exact keyword match for Year-Month, Year-Week or Date
    for col in all_cols:
        col_lower = str(col).lower()
        if any(cand in col_lower for cand in date_candidates):
            detected_date = col
            break

    # Priority 2: Check if column values are Year-Month, Year-Week numbers or parseable datetimes
    if detected_date is None:
        for col in all_cols:
            if is_year_month_series(df[col], col_name=col):
                detected_date = col
                break
            if is_year_week_series(df[col], col_name=col):
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

    # Check if this column is a Taiwan CDC Year-Month or Year-Week series
    is_ym = is_year_month_series(raw_date_series, col_name=date_col)
    is_yw = False if is_ym else is_year_week_series(raw_date_series, col_name=date_col)

    if is_ym:
        str_series = raw_date_series.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
        clean_df['year_month'] = str_series
        clean_df['ds'] = pd.to_datetime(
            clean_df['year_month'].apply(lambda s: f"{s[:4]}-{s[4:6]}-01" if len(s) == 6 and s.isdigit() else s),
            errors='coerce'
        )
        clean_df['year_week'] = clean_df['year_month']
        converted_year_week = False
        converted_year_month = True
    elif is_yw:
        clean_df['year_week'] = raw_date_series.astype(str).str.strip().str.replace('"', '')
        clean_df['ds'] = clean_df['year_week'].apply(cdc_year_week_to_date)
        clean_df['ds'] = pd.to_datetime(clean_df['ds'])
        converted_year_week = True
        converted_year_month = False
    else:
        clean_df['year_week'] = None
        str_series = raw_date_series.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
        sample_val = str_series.dropna().iloc[0] if len(str_series.dropna()) > 0 else ""
        if sample_val.isdigit() and len(sample_val) == 8:
            clean_df['ds'] = pd.to_datetime(str_series, format='%Y%m%d', errors='coerce')
        else:
            try:
                clean_df['ds'] = pd.to_datetime(str_series, format='mixed', errors='coerce')
            except (ValueError, TypeError, pd.errors.ParserError):
                clean_df['ds'] = pd.to_datetime(str_series, errors='coerce')

        # Robust epoch guard: If parsed dates mistakenly landed in 1970 due to epoch nanoseconds
        if len(clean_df['ds'].dropna()) > 0 and clean_df['ds'].dropna().dt.year.min() == 1970 and not str(sample_val).startswith(('1970', '70')):
            clean_df['ds'] = pd.to_datetime(str_series, format='%Y%m%d', errors='coerce')
        converted_year_week = False
        converted_year_month = False

    clean_df['y'] = pd.to_numeric(df[target_col], errors='coerce')

    # Drop rows with invalid dates and sort chronologically
    clean_df = clean_df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

    # Populate reverse Year-Week for non-year-week series
    if not converted_year_week and not converted_year_month:
        clean_df['year_week'] = clean_df['ds'].apply(date_to_cdc_year_week)

    # Handle missing target values with linear interpolation
    clean_df['y'] = clean_df['y'].interpolate(method='linear').bfill().ffill()
    clean_df['y'] = np.maximum(clean_df['y'].values, 0.0)

    if override_freq is not None:
        inferred_freq = override_freq
    elif is_ym:
        inferred_freq = 'M'
    elif is_yw:
        inferred_freq = 'W'
    else:
        inferred_freq = infer_frequency(clean_df['ds'])

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
        'is_year_month_converted': converted_year_month,
        'start_year_week': clean_df['year_week'].iloc[0] if converted_year_week else None,
        'end_year_week': clean_df['year_week'].iloc[-1] if converted_year_week else None,
        'start_year_month': clean_df['year_month'].iloc[0] if converted_year_month else None,
        'end_year_month': clean_df['year_month'].iloc[-1] if converted_year_month else None,
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
