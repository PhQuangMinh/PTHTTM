import os
import json
import math
import unicodedata
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
import tensorflow as tf

try:
    import plotly.express as px
except Exception:
    px = None

# Prevent concurrent TF/Keras calls during Streamlit reruns/sessions
_PREDICT_LOCK = threading.Lock()

# -----------------------------
# Paths and constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'weather-vn.csv')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'models')
CONFIG_PATH = os.path.join(ARTIFACT_DIR, 'config.json')
SCALERS_PATH = os.path.join(ARTIFACT_DIR, 'scalers.joblib')
RNN_CKPT = os.path.join(ARTIFACT_DIR, 'rnn_vn_weather.keras')
LSTM_CKPT = os.path.join(ARTIFACT_DIR, 'lstm_vn_weather.keras')
LOCAL_VN_GEOJSON_PATH = os.path.join(BASE_DIR, 'vn.json')

# Public Vietnam provinces GeoJSON. We try multiple mirrors because some networks block raw GitHub.
VN_GEOJSON_URLS = [
    # CDN mirror (often more reliable than raw GitHub)
    'https://cdn.jsdelivr.net/gh/gia-ha/Vietnam-GeoJSON@master/Provinces/VN-provinces.json',
    # Raw GitHub fallback
    'https://raw.githubusercontent.com/gia-ha/Vietnam-GeoJSON/master/Provinces/VN-provinces.json',
    # Alternative raw mirror (may or may not work depending on network)
    'https://raw.fastgit.org/gia-ha/Vietnam-GeoJSON/master/Provinces/VN-provinces.json',
]

# Province -> region mapping (same as notebook)
PROVINCE_TO_REGION3 = {
    # North
    'ha noi': 'North', 'hai phong': 'North', 'quang ninh': 'North', 'hai duong': 'North',
    'bac ninh': 'North', 'bac giang': 'North', 'lang son': 'North', 'lao cai': 'North',
    'yen bai': 'North', 'son la': 'North', 'hoa binh': 'North', 'ha giang': 'North',
    'tuyen quang': 'North', 'phu tho': 'North', 'thai nguyen': 'North', 'nam dinh': 'North',
    'thai binh': 'North', 'ninh binh': 'North', 'bac kan': 'North', 'dien bien': 'North',
    'cao bang': 'North', 'lai chau': 'North', 'vinh phuc': 'North', 'quang ninh': 'North',
    # Central
    'thua thien hue': 'Central', 'da nang': 'Central', 'quang nam': 'Central', 'quang ngai': 'Central',
    'binh dinh': 'Central', 'phu yen': 'Central', 'khanh hoa': 'Central', 'ninh thuan': 'Central',
    'binh thuan': 'Central', 'quang tri': 'Central', 'quang binh': 'Central', 'ha tinh': 'Central',
    'nghe an': 'Central', 'thanh hoa': 'Central', 'dak lak': 'Central', 'dak nong': 'Central',
    'gia lai': 'Central', 'kon tum': 'Central', 'lam dong': 'Central',
    # South
    'ho chi minh': 'South', 'can tho': 'South', 'dong nai': 'South', 'ba ria-vung tau': 'South',
    'ba ria - vung tau': 'South', 'binh duong': 'South', 'binh phuoc': 'South', 'tay ninh': 'South',
    'long an': 'South', 'tien giang': 'South', 'ben tre': 'South', 'tra vinh': 'South',
    'vinh long': 'South', 'dong thap': 'South', 'an giang': 'South', 'kien giang': 'South',
    'hau giang': 'South', 'soc trang': 'South', 'bac lieu': 'South', 'ca mau': 'South'
}

DATE_COL_CANDIDATES = ['date', 'time', 'datetime', 'timestamp', 'day']
PROVINCE_COL_CANDIDATES = ['province', 'tinh', 'city', 'station', 'location', 'site']

NAME_ALIASES = {
    'temperature': ['temperature', 'temp', 'tavg', 'tmean'],
    'humidity': ['humidity', 'rh'],
    'rainfall': ['rainfall', 'precip', 'precipitation', 'rain'],
    'wind_speed': ['wind_speed', 'wind', 'windspd']
}

# -----------------------------
# Utilities
# -----------------------------

def normalize_name(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = s.replace('tp.', '').replace('tp', '').replace('tinh', '').replace('thanh pho', '')
    s = s.replace('-', ' ').replace('_', ' ').replace('  ', ' ').strip()
    return s


def detect_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    return None


@st.cache_data(show_spinner=False)
def load_geojson(url: str) -> Optional[dict]:
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={
                # Some hosts block requests without UA
                'User-Agent': 'PTHTTM-weather-app/1.0'
            },
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def load_geojson_any(urls: Tuple[str, ...]) -> Optional[dict]:
    for url in urls:
        gj = load_geojson(url)
        if gj and isinstance(gj, dict) and gj.get('type') in ('FeatureCollection', 'Topology'):
            return gj
    return None


@st.cache_data(show_spinner=False)
def load_geojson_file(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            gj = json.load(f)
        if isinstance(gj, dict) and gj.get('type') in ('FeatureCollection', 'Topology'):
            return gj
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def geocode_provinces_osm(provinces: Tuple[str, ...]) -> pd.DataFrame:
    """
    Fallback when GeoJSON cannot be downloaded: use OSM Nominatim to get lat/lon per province.
    Cached to avoid rate limiting across reruns.
    """
    rows = []
    url = 'https://nominatim.openstreetmap.org/search'
    headers = {'User-Agent': 'PTHTTM-weather-app/1.0'}

    for p in provinces:
        q = f'{p}, Vietnam'
        lat, lon = None, None
        try:
            r = requests.get(url, params={'q': q, 'format': 'json', 'limit': 1}, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json() or []
                if isinstance(data, list) and len(data) > 0:
                    lat = float(data[0].get('lat'))
                    lon = float(data[0].get('lon'))
        except Exception:
            lat, lon = None, None

        rows.append({'province': p, 'lat': lat, 'lon': lon})

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    # Reset TF/Keras global state to avoid rare name_scope stack issues across reruns
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    scalers = joblib.load(SCALERS_PATH)
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    rnn_model = tf.keras.models.load_model(RNN_CKPT, compile=False) if os.path.exists(RNN_CKPT) else None
    lstm_model = tf.keras.models.load_model(LSTM_CKPT, compile=False) if os.path.exists(LSTM_CKPT) else None
    return config, feature_scaler, target_scaler, rnn_model, lstm_model


@st.cache_data(show_spinner=True)
def load_and_prepare_data(csv_path: str,
                         target_cols_required: List[str]) -> Tuple[pd.DataFrame, str, List[str]]:
    df_raw = pd.read_csv(csv_path, low_memory=False)
    # Detect columns
    date_col = detect_first_existing(df_raw, DATE_COL_CANDIDATES)
    if date_col is None:
        raise ValueError('Không tìm thấy cột thời gian trong CSV.')
    prov_col = detect_first_existing(df_raw, PROVINCE_COL_CANDIDATES)
    if prov_col is None:
        raise ValueError('Không tìm thấy cột tỉnh/thành (province/city) trong CSV.')

    # Map target columns
    lower_map = {c.lower(): c for c in df_raw.columns}
    standard_cols = {}
    for std, aliases in NAME_ALIASES.items():
        for a in aliases:
            if a in lower_map:
                standard_cols[std] = lower_map[a]
                break

    # Take only variables present that model expects
    target_cols = [c for c in target_cols_required if c in standard_cols.keys()]
    if len(target_cols) == 0:
        raise ValueError('Không tìm thấy biến mục tiêu phù hợp trong CSV.')

    # Build working df
    work_map = {'date': date_col, 'province': prov_col}
    for t in target_cols:
        work_map[t] = standard_cols[t]
    df = df_raw[list(work_map.values())].copy()
    df.columns = list(work_map.keys())

    # Types and sort
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Region mapping from province name
    df['province_norm'] = df['province'].astype(str).map(normalize_name)
    df['region3'] = df['province_norm'].map(PROVINCE_TO_REGION3).fillna('Unknown')

    # Aggregate to daily per province
    daily = []
    for p in df['province'].unique():
        d = df[df['province'] == p].set_index('date').sort_index()
        d_daily = d.resample('D').mean(numeric_only=True)
        d_daily['province'] = p
        d_daily['province_norm'] = normalize_name(p)
        d_daily['region3'] = d['region3'].iloc[0]
        # Interpolate and fill
        for c in target_cols:
            if c in d_daily.columns:
                d_daily[c] = d_daily[c].interpolate(method='time', limit_direction='both')
                d_daily[c] = d_daily[c].fillna(d_daily[c].median())
        daily.append(d_daily.reset_index())
    proc = pd.concat(daily, ignore_index=True)
    proc = proc.sort_values(['province', 'date']).reset_index(drop=True)

    return proc, prov_col, target_cols


def build_window_for_province(proc_df: pd.DataFrame,
                              province: str,
                              lookback: int,
                              region_feature_cols: List[str],
                              target_cols: List[str],
                              feature_scaler,
                              allow_pad: bool = False) -> Optional[np.ndarray]:
    d_all = proc_df[proc_df['province'] == province].sort_values('date')
    if d_all.empty:
        return None
    d = d_all.tail(lookback)
    if len(d) < lookback and not allow_pad:
        return None
    # one-hot region (align columns)
    region = d_all['region3'].iloc[-1]
    ohe = {col: 0.0 for col in region_feature_cols}
    key = f"region_{region}"
    if key in ohe:
        ohe[key] = 1.0
    ohe_row = np.array([ohe[c] for c in region_feature_cols], dtype='float32')

    # numeric part: past targets
    xnum = d[target_cols].astype('float32').values
    xnum_s = feature_scaler.transform(xnum)

    # pad if needed
    if len(d) < lookback and allow_pad and len(d) > 0:
        pad_rows = lookback - len(d)
        first_row = xnum_s[0:1, :]
        pad_block = np.repeat(first_row, pad_rows, axis=0)
        xnum_s = np.vstack([pad_block, xnum_s])

    feats = np.hstack([np.tile(ohe_row, (lookback, 1)), xnum_s]).astype('float32')
    return feats[np.newaxis, ...]  # (1, lookback, feature_dim)


def iterative_forecast(window_feats: np.ndarray,
                       steps: int,
                       model,
                       feature_scaler,
                       target_scaler,
                       region_feature_cols: List[str],
                       target_cols: List[str]) -> pd.DataFrame:
    # window_feats: (1, lookback, feature_dim) where feature_dim = len(region_onehot) + len(target_cols)
    lookback = window_feats.shape[1]
    region_onehot = window_feats[0, 0, :len(region_feature_cols)]  # constant over time
    x_hist = window_feats.copy()  # will roll forward with predictions

    preds = []
    for _ in range(steps):
        # Avoid model.predict() in Streamlit reruns (can trigger Keras name_scope stack issues).
        with _PREDICT_LOCK:
            y_scaled_t = model(x_hist, training=False)
        y_scaled = y_scaled_t.numpy() if hasattr(y_scaled_t, "numpy") else np.asarray(y_scaled_t)
        # inverse target
        try:
            y = target_scaler.inverse_transform(y_scaled)
        except Exception:
            y = y_scaled
        preds.append(y[0])
        # prepare next step: append predicted numeric feat (after feature scaling) and drop first
        try:
            y_num_for_feat = feature_scaler.transform(y)
        except Exception:
            y_num_for_feat = y
        next_row = np.hstack([region_onehot, y_num_for_feat[0]]).astype('float32')
        x_hist = np.concatenate([x_hist[:, 1:, :], next_row.reshape(1, 1, -1)], axis=1)

    df_pred = pd.DataFrame(preds, columns=target_cols)
    return df_pred


# ---- Helpers to compute province centroid (for Plotly Mapbox points) ----
def _centroid_from_coords(coords: List) -> Tuple[float, float]:
    # coords: list of [lon, lat]
    lons, lats = [], []
    for c in coords:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            lons.append(float(c[0]))
            lats.append(float(c[1]))
    if not lons:
        return 106.0, 16.0
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def compute_province_centroids(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson.get('features', []):
        props = feat.get('properties', {}) or {}
        name = props.get('NAME_1') or props.get('name') or props.get('ten_tinh') or ''
        name_norm = normalize_name(name)
        geom = feat.get('geometry', {}) or {}
        gtype = geom.get('type')
        coords = geom.get('coordinates', [])
        lon, lat = 106.0, 16.0
        try:
            if gtype == 'Polygon':
                # coords[0] exterior ring
                lon, lat = _centroid_from_coords(coords[0] if coords else [])
            elif gtype == 'MultiPolygon':
                # flatten all rings
                all_points = []
                for poly in coords:
                    if poly and isinstance(poly, list):
                        all_points.extend(poly[0])
                lon, lat = _centroid_from_coords(all_points)
        except Exception:
            lon, lat = 106.0, 16.0
        rows.append({'province_name': name, 'province_norm': name_norm, 'lon': lon, 'lat': lat})
    return pd.DataFrame(rows)


def _metric_label(metric: str) -> str:
    m = (metric or '').lower()
    if m == 'temperature':
        return 'Nhiệt độ (°C)'
    if m == 'humidity':
        return 'Độ ẩm (%)'
    if m == 'rainfall':
        return 'Lượng mưa'
    if m == 'wind_speed':
        return 'Tốc độ gió (m/s)'
    return metric


def _metric_scale(metric: str) -> str:
    m = (metric or '').lower()
    if m == 'temperature':
        return 'Reds'
    if m == 'humidity':
        return 'Blues'
    if m == 'rainfall':
        return 'Viridis'
    if m == 'wind_speed':
        return 'Turbo'
    return 'Viridis'


def _format_metric_value(metric: str, v: float) -> str:
    if pd.isna(v):
        return ''
    m = (metric or '').lower()
    if m == 'temperature':
        return f'{v:.1f}°C'
    if m == 'humidity':
        return f'{v:.0f}%'
    if m == 'wind_speed':
        return f'{v:.2f} m/s'
    if m == 'rainfall':
        return f'{v:.2f}'
    return f'{v:.2f}'


def _make_marker_sizes(values: pd.Series, min_size: float = 8.0, max_size: float = 24.0) -> pd.Series:
    v = pd.to_numeric(values, errors='coerce')
    if v.notna().sum() <= 1:
        return pd.Series(np.full(len(v), (min_size + max_size) / 2), index=v.index)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return pd.Series(np.full(len(v), (min_size + max_size) / 2), index=v.index)
    scaled = (v - vmin) / (vmax - vmin)
    return (min_size + scaled * (max_size - min_size)).fillna((min_size + max_size) / 2)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='Dự báo thời tiết Việt Nam (RNN/LSTM)', layout='wide')
st.title('Dự báo thời tiết Việt Nam bằng RNN & LSTM')

# Force light theme (white background, black text)
st.markdown(
    """
<style>
/* App background */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: #ffffff !important;
  color: #111111 !important;
}
/* Common text */
html, body, p, span, label, div, h1, h2, h3, h4, h5, h6 {
  color: #111111 !important;
}
/* Inputs: force white background + black border/text (selectbox, date_input, etc.) */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
  background: #ffffff !important;
  color: #111111 !important;
  border-color: #111111 !important;
}
/* Streamlit buttons/toggles */
button, [role="button"] {
  border-color: #111111 !important;
}
/* Popovers (date picker) */
[data-baseweb="popover"] * {
  color: #111111 !important;
}
[data-baseweb="popover"] {
  background: #ffffff !important;
  border: 1px solid #111111 !important;
}
/* Tables */
[data-testid="stDataFrame"] * {
  color: #111111 !important;
}
</style>
""",
    unsafe_allow_html=True
)

# Defaults (no left config panel)
MAX_FORECAST_DAYS = 7
allow_pad = True

# Load artifacts
try:
    config, feature_scaler, target_scaler, rnn_model, lstm_model = load_artifacts()
except Exception as e:
    st.error(f'Lỗi nạp mô hình/scaler: {e}')
    st.stop()

cfg_lookback = int(config.get('lookback_days', 30))
region_feature_cols = list(config.get('region_feature_cols', []))
target_cols = list(config.get('target_cols', []))

# Fixed lookback (from training config)
lookback = int(cfg_lookback or 30)

# Data source (fixed)
csv_path = DATA_CSV

# Load and preprocess
try:
    proc_df, province_col, target_cols_present = load_and_prepare_data(csv_path, target_cols)
except Exception as e:
    st.error(f'Lỗi xử lý dữ liệu: {e}')
    st.stop()

# Province list and controls
provinces = sorted(proc_df['province'].unique().tolist())

# Controls above the map (no sidebar)
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
with ctrl1:
    model_choice = st.selectbox('Chọn mô hình', ['LSTM', 'RNN'], index=0)
with ctrl2:
    global_last_date = proc_df['date'].max()
    min_date = (global_last_date + timedelta(days=1)).date()
    max_date = (global_last_date + timedelta(days=MAX_FORECAST_DAYS)).date()
    selected_date = st.date_input(
        'selected_date',
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        help=f'Chọn ngày cần dự báo (tối đa {MAX_FORECAST_DAYS} ngày tới).'
    )
with ctrl3:
    show_labels = st.checkbox('Hiển thị nhãn trên bản đồ', value=True)

# Confirm models
model = lstm_model if model_choice == 'LSTM' else rnn_model
if model is None:
    st.error(f'Không tìm thấy checkpoint cho mô hình {model_choice}. Hãy huấn luyện và lưu trước.')
    st.stop()

st.markdown('---')
st.subheader('Dự báo toàn quốc trên bản đồ (theo tỉnh)')

# Forecast only the selected_date for all provinces
forecast_date = pd.Timestamp(selected_date)
pred_rows = []
for p in provinces:
    win = build_window_for_province(
        proc_df, p, lookback, region_feature_cols, target_cols_present, feature_scaler, allow_pad=allow_pad
    )
    if win is None:
        continue

    last_date_p = proc_df.loc[proc_df['province'] == p, 'date'].max()
    if pd.isna(last_date_p):
        continue

    step = int((forecast_date.normalize() - pd.Timestamp(last_date_p).normalize()).days)
    if step <= 0:
        continue

    # Safety: selected_date is constrained by max_date, but keep a guard
    if step > int(MAX_FORECAST_DAYS):
        continue

    df_pred = iterative_forecast(win, step, model, feature_scaler, target_scaler, region_feature_cols, target_cols_present)
    if df_pred.empty:
        continue
    y = df_pred.iloc[-1].to_dict()
    y['date'] = forecast_date
    y['province'] = p
    y['province_norm'] = normalize_name(p)
    pred_rows.append(y)

if len(pred_rows) == 0:
    st.warning('Không đủ dữ liệu để dự báo cho bất kỳ tỉnh nào. Hãy giảm lookback hoặc bật chế độ đệm.')
    # Diagnostics: số ngày có dữ liệu theo tỉnh
    diag = proc_df.groupby('province').agg(days_available=('date', 'nunique'),
                                           last_date=('date', 'max')).reset_index()
    diag['cần_thêm_ngày'] = (lookback - diag['days_available']).clip(lower=0)
    st.markdown('#### Số ngày dữ liệu theo tỉnh')
    diag_tbl = diag.sort_values('days_available').reset_index(drop=True)
    diag_styled = (
        diag_tbl.style
        .set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border': '1px solid black',
        })
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', 'white'), ('color', 'black'), ('border', '1px solid black')]},
            {'selector': 'td', 'props': [('border', '1px solid black')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '1px solid black')]},
        ])
    )
    st.dataframe(diag_styled, use_container_width=True)
    st.stop()

pred_latest = pd.DataFrame(pred_rows)
map_date = forecast_date

# Optional detail table selection (kept outside left panel)
sample_provs = st.multiselect('Xem chi tiết dự báo cho các tỉnh', provinces[:10], default=provinces[:5])

# Load GeoJSON (preferred) -> compute centroids; fallback: OSM geocode per province.
if px is None:
    st.error("Thiếu thư viện Plotly. Hãy cài: `pip install plotly` để hiển thị bản đồ Mapbox.")
else:
    geojson = None
    if geojson is None and os.path.exists(LOCAL_VN_GEOJSON_PATH):
        geojson = load_geojson_file(LOCAL_VN_GEOJSON_PATH)

    if geojson is None:
        geojson = load_geojson_any(tuple(VN_GEOJSON_URLS))

    pts = pred_latest.copy()
    pts['lat'] = np.nan
    pts['lon'] = np.nan

    if geojson:
        centroids = compute_province_centroids(geojson)
        pts = pts.merge(centroids[['province_norm', 'lat', 'lon']], on='province_norm', how='left', suffixes=('', '_gj'))
        # Prefer GeoJSON coords when present
        pts['lat'] = pts['lat_gj']
        pts['lon'] = pts['lon_gj']
        pts = pts.drop(columns=['lat_gj', 'lon_gj'])
    else:
        st.info('Không tải được GeoJSON từ các nguồn online. Thử lấy tọa độ tỉnh từ OpenStreetMap...')

    # If still missing coords, try OSM geocoding for missing provinces
    if pts['lat'].isna().any() or pts['lon'].isna().any():
        prov_need = tuple(sorted(pts.loc[pts['lat'].isna() | pts['lon'].isna(), 'province'].unique().tolist()))
        if prov_need:
            geo_pts = geocode_provinces_osm(prov_need)
            pts = pts.merge(geo_pts, on='province', how='left', suffixes=('', '_osm'))
            pts['lat'] = pts['lat'].fillna(pts['lat_osm'])
            pts['lon'] = pts['lon'].fillna(pts['lon_osm'])
            pts = pts.drop(columns=['lat_osm', 'lon_osm'])

    pts = pts.dropna(subset=['lat', 'lon']).copy()

    if pts.empty:
        st.info('Không lấy được tọa độ tỉnh để vẽ bản đồ. Hiển thị bảng kết quả bên dưới.')
    else:
        # Always show 4 common weather metrics if present (as tabs), like the mock in test_mapbox.py
        preferred_metrics = ['temperature', 'humidity', 'rainfall', 'wind_speed']
        metrics = [m for m in preferred_metrics if m in target_cols_present]
        if not metrics:
            metrics = target_cols_present[:1]

        missing = [m for m in preferred_metrics if m not in target_cols_present]
        if missing:
            st.caption('Thiếu biến trong dữ liệu/model: ' + ', '.join(missing))

        tabs = st.tabs([_metric_label(m) for m in metrics])
        for tab, metric in zip(tabs, metrics):
            with tab:
                pts['_metric_'] = pd.to_numeric(pts[metric], errors='coerce')
                pts['_marker_size_'] = _make_marker_sizes(pts['_metric_'])
                pts['_label_'] = pts['_metric_'].map(lambda v: _format_metric_value(metric, v))

                hover_fmt = {
                    'date': True,
                    'province': True,
                    'lat': False,
                    'lon': False,
                    '_marker_size_': False,
                    '_label_': False,
                    '_metric_': False,
                }
                for m2 in preferred_metrics:
                    if m2 in pts.columns:
                        if m2 == 'temperature':
                            hover_fmt[m2] = ':.1f'
                        elif m2 == 'humidity':
                            hover_fmt[m2] = ':.0f'
                        elif m2 == 'wind_speed':
                            hover_fmt[m2] = ':.2f'
                        else:
                            hover_fmt[m2] = ':.2f'

                fig = px.scatter_mapbox(
                    pts,
                    lat='lat',
                    lon='lon',
                    color='_metric_',
                    size='_marker_size_',
                    size_max=26,
                    zoom=4.5,
                    hover_name='province',
                    hover_data=hover_fmt,
                    color_continuous_scale=_metric_scale(metric),
                    height=650,
                )
                fig.update_layout(
                    mapbox_style='carto-positron',
                    margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color='black',
                    coloraxis_colorbar=dict(
                        title=_metric_label(metric),
                        tickfont=dict(color='black'),
                        outlinecolor='black',
                        outlinewidth=1,
                        bgcolor='rgba(255,255,255,0.95)',
                    ),
                )

                if show_labels:
                    fig.update_traces(
                        text=pts['_label_'],
                        textposition='top center',
                        textfont={'size': 14, 'color': 'black'}
                    )

                st.plotly_chart(fig, use_container_width=True)

# Show table
st.markdown(f'#### Bảng dự báo cho ngày {map_date.date()}')
show_cols = ['province', 'date'] + target_cols_present
tbl = pred_latest[show_cols].sort_values('province').reset_index(drop=True)
tbl_styled = (
    tbl.style
    .set_properties(**{
        'background-color': 'white',
        'color': 'black',
        'border': '1px solid black',
    })
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'white'), ('color', 'black'), ('border', '1px solid black')]},
        {'selector': 'td', 'props': [('border', '1px solid black')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '1px solid black')]},
    ])
)
st.dataframe(tbl_styled, use_container_width=True)

# Details for selected provinces (multi-day)
if sample_provs:
    st.markdown('---')
    st.subheader('Chi tiết dự báo theo tỉnh (ngày đã chọn)')
    for p in sample_provs:
        dfp = pred_latest[pred_latest['province'] == p].sort_values('date')
        if dfp.empty:
            continue
        st.markdown(f'**{p}**')
        sub_tbl = dfp[['date'] + target_cols_present].reset_index(drop=True)
        sub_styled = (
            sub_tbl.style
            .set_properties(**{
                'background-color': 'white',
                'color': 'black',
                'border': '1px solid black',
            })
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', 'white'), ('color', 'black'), ('border', '1px solid black')]},
                {'selector': 'td', 'props': [('border', '1px solid black')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '1px solid black')]},
            ])
        )
        st.dataframe(sub_styled, use_container_width=True)

st.caption('Lưu ý: Mô hình dự báo theo cửa sổ trượt đã huấn luyện, kết quả phụ thuộc dữ liệu đầu vào và scaler đã lưu.')

# Congratulations! You’ve won a $1,000 Amazon gift card. Click the link below and confirm your details now!
# URGENT: Your bank account has been suspended. Verify your information immediately to avoid permanent closure.
# Hey, are we still meeting at 7 pm tonight? Let me know when you’re on the way.