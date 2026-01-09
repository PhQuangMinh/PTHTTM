import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

# Cấu hình port mong muốn (8505) nếu chạy bằng `streamlit run`
os.environ.setdefault("STREAMLIT_SERVER_PORT", "8505")

# =============================
# Paths and constants
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SCALER_DIR = os.path.join(BASE_DIR, "scaler")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TICKERS = ["BID", "BVH", "CTG"]
MODEL_TYPES = ["LSTM", "RNN"]

DATE_COLS = [
    "date",
    "time",
    "datetime",
    "ngay",
    "timestamp",
    "Date",
    "Datetime",
    "DTYYYYMMDD",  # phổ biến trong dữ liệu VNDirect
]
PRICE_COLS = [
    "close",
    "adj close",
    "close_price",
    "close price",
    "gia dong cua",
    "Close",
    "closeprice",
]


# =============================
# Utilities
# =============================
def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Tìm cột theo tên ứng viên, cho phép dữ liệu có ký tự lạ như <> , khoảng trắng...
    So khớp dựa trên tên đã chuẩn hóa (chỉ còn a-z0-9).
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    normalized_to_original = {norm(col): col for col in df.columns}

    # Thử khớp chính xác theo normalize
    for cand in candidates:
        cand_n = norm(cand)
        if cand_n in normalized_to_original:
            return normalized_to_original[cand_n]

    # Fallback: tìm cột có chứa từ khóa ứng viên (ví dụ: 'date' nằm trong 'dtyyyymmdd')
    for cand in candidates:
        cand_n = norm(cand)
        for col_n, col in normalized_to_original.items():
            if cand_n in col_n or col_n in cand_n:
                return col

    return None


def load_ticker_data(ticker: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load CSV for a ticker and return (dates, close_prices).
    """
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = detect_col(df, DATE_COLS)
    price_col = detect_col(df, PRICE_COLS)
    if date_col is None or price_col is None:
        raise ValueError("Không phát hiện cột thời gian/giá trong dữ liệu")

    # Chuẩn hóa cột ngày: nếu là số 8 chữ số (YYYYMMDD) thì parse theo định dạng
    date_series = df[date_col]
    if pd.api.types.is_numeric_dtype(date_series) or date_series.astype(str).str.fullmatch(r"\d{8}").all():
        df[date_col] = pd.to_datetime(date_series.astype(str), format="%Y%m%d", errors="coerce")
    else:
        df[date_col] = pd.to_datetime(date_series, errors="coerce")

    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    return df[date_col].reset_index(drop=True), df[price_col].reset_index(drop=True)


def load_scaler(ticker: str):
    pkl = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
    if os.path.exists(pkl):
        return joblib.load(pkl)
    # fallback if not saved
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler((0, 1))


def load_model(ticker: str, model_type: str, prefer_best: bool = True):
    mt = "lstm" if model_type.upper() == "LSTM" else "rnn"
    best = os.path.join(MODELS_DIR, f"{ticker}_{mt}_best.keras")
    final = os.path.join(MODELS_DIR, f"{ticker}_{mt}_final.keras")
    ckpt = best if (prefer_best and os.path.exists(best)) else (final if os.path.exists(final) else None)
    if ckpt is None:
        raise FileNotFoundError(f"Không tìm thấy model: {best} hoặc {final}")
    return tf.keras.models.load_model(ckpt), ckpt


def make_sequences(series_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given scaled series shaped (N, 1), build X (N-lookback, lookback, 1) and y (N-lookback, 1)
    for 1-step ahead forecasting.
    """
    X, y = [], []
    arr = series_scaled.reshape(-1, 1)
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback : i, 0])
        y.append(arr[i, 0])
    if not X:
        return np.empty((0, lookback, 1), dtype="float32"), np.empty((0, 1), dtype="float32")
    X = np.array(X, dtype="float32").reshape(-1, lookback, 1)
    y = np.array(y, dtype="float32").reshape(-1, 1)
    return X, y


def last_n_predictions(
    dates: pd.Series, prices: pd.Series, scaler, model, lookback: int, last_n: int = 30
) -> pd.DataFrame:
    """
    Build sliding 1-step predictions across the series, then return last_n rows with
    columns: ['date', 'actual', 'pred'].
    """
    values = prices.values.astype("float32").reshape(-1, 1)
    # Fit scaler if needed
    try:
        values_scaled = scaler.transform(values)
    except Exception:
        scaler.fit(values)
        values_scaled = scaler.transform(values)

    X, y = make_sequences(values_scaled, lookback)
    if len(X) == 0:
        return pd.DataFrame(columns=["date", "actual", "pred"])

    y_pred_scaled = model.predict(X, verbose=0)
    # Inverse scale
    try:
        y_true = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred_scaled)
    except Exception:
        y_true, y_pred = y, y_pred_scaled

    target_dates = dates.iloc[lookback:].reset_index(drop=True)
    df = pd.DataFrame(
        {
            "date": target_dates[-last_n:],
            "actual": y_true.ravel()[-last_n:],
            "pred": y_pred.ravel()[-last_n:],
        }
    )
    return df


def mae_rmse(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    mae = float(np.mean(np.abs(a - b)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return mae, rmse


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Dự báo cổ phiếu (Local)", page_icon="📈", layout="wide")

# Simple white theme and readable text
st.markdown(
    """
    <style>
    .main, .stApp { background: #FFFFFF !important; color: #111111 !important; }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { background: #FFFFFF !important; color: #111111 !important; }
    .stMetric { background: #fff; color: #111111; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #111111 !important; }
    .stSelectbox div, .stSlider, .stSlider * { color: #111111 !important; }
    input, textarea, select, option { color: #111111 !important; background: #FFFFFF !important; }
    /* DataFrame bảng trắng chữ đen, viền xám đậm */
    [data-testid="stDataFrame"] table { background: #FFFFFF !important; color: #111111 !important; }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { color: #111111 !important; border: 1px solid #222 !important; }
    [data-testid="stDataFrame"] thead { background: #f5f5f5 !important; color: #111111 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Ứng dụng dự báo cổ phiếu (LSTM/RNN)")
st.caption("Chọn sàn, chọn mô hình và xem đường dự báo 30 ngày gần nhất.")

with st.sidebar:
    st.header("Cấu hình")
    ticker = st.selectbox("Chọn sàn/mã", TICKERS, index=0)
    model_type = st.selectbox("Chọn mô hình", MODEL_TYPES, index=0)
    lookback = st.slider("Cửa sổ lookback", min_value=10, max_value=120, value=30, step=5)
    prefer_best = st.checkbox("Ưu tiên model best (checkpoint)", value=True)
    st.info("Biểu đồ cố định hiển thị 30 ngày cuối cùng trong dữ liệu.")

# Load data
try:
    dates, prices = load_ticker_data(ticker)
except Exception as e:
    st.error(f"Lỗi nạp dữ liệu {ticker}: {e}")
    st.stop()

# Load scaler
try:
    scaler = load_scaler(ticker)
except Exception as e:
    st.warning(f"Lỗi nạp scaler {ticker}: {e}. Sẽ fit trực tiếp trên dữ liệu hiện có.")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler((0, 1))

# Load model
try:
    model, ckpt_path = load_model(ticker, model_type, prefer_best=prefer_best)
except Exception as e:
    st.error(f"Lỗi nạp mô hình {ticker} ({model_type}): {e}")
    st.stop()

# Inference for last 30 days
LAST_N = 30
df_plot = last_n_predictions(dates, prices, scaler, model, lookback=lookback, last_n=LAST_N)
if df_plot.empty:
    st.warning("Dữ liệu không đủ để tạo chuỗi lookback. Hãy giảm lookback hoặc kiểm tra dữ liệu.")
    st.stop()

mae, rmse = mae_rmse(df_plot["actual"].values, df_plot["pred"].values)
# Cố định trục Y 0-100, tick mỗi 10 để zoom gần và đồng nhất báo cáo
y_range = [0, 100]

# Top KPIs
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("MAE", f"{mae:.3f}")
with col2:
    st.metric("RMSE", f"{rmse:.3f}")
with col3:
    st.write(f"Model: `{os.path.basename(ckpt_path)}`")

# Line chart Actual vs Predict
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_plot["date"],
        y=df_plot["actual"],
        mode="lines",
        name="Giá thực tế",
        line=dict(color="#2E86DE", width=2),
        hovertemplate="Ngày: %{x|%a %d %b %Y}<br>Giá thực tế: %{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=df_plot["date"],
        y=df_plot["pred"],
        mode="lines",
        name="Giá dự đoán",
        line=dict(color="#54a0ff", width=2, shape="spline"),
        hovertemplate="Ngày: %{x|%a %d %b %Y}<br>Giá dự đoán: %{y:.2f}<extra></extra>",
    )
)
fig.update_layout(
    title=f"{ticker} - {model_type} | 30 ngày gần nhất",
    xaxis_title="Ngày",
    yaxis_title="Giá",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="black")),
    margin=dict(l=10, r=10, t=60, b=10),
    template="simple_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    hovermode="x unified",
    font=dict(color="black")
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="#eaeaea",
    tickformat="%a %d",
    tickformatstops=[
        dict(dtickrange=[None, 86400000 * 31], value="%a %d"),
        dict(dtickrange=[86400000 * 31, None], value="%b %Y"),
    ],
    tickfont=dict(color="black"),
    title_font=dict(color="black"),
    linecolor="black",
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="#eaeaea",
    tickfont=dict(color="black"),
    title_font=dict(color="black"),
    linecolor="black",
    dtick=10,
    range=y_range,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Bảng dữ liệu 30 ngày cuối"):
    st.dataframe(df_plot, use_container_width=True)

st.caption(
    "Mẹo: Nếu không thấy đường dự báo, hãy giảm lookback hoặc đảm bảo scaler/model trùng với thiết lập khi train."
)

