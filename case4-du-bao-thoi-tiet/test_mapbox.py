import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# =============================
# Mock data: tọa độ tỉnh/thành
# =============================
PROVINCES = [
    ("Hà Nội", 21.0285, 105.8542),
    ("Hải Phòng", 20.8449, 106.6881),
    ("Quảng Ninh", 21.0064, 107.2925),
    ("Nam Định", 20.4388, 106.1621),
    ("Thanh Hóa", 19.8067, 105.7852),
    ("Nghệ An", 18.6796, 105.6813),
    ("Huế", 16.4637, 107.5909),
    ("Đà Nẵng", 16.0544, 108.2022),
    ("Khánh Hòa", 12.2585, 109.0526),
    ("TP. Hồ Chí Minh", 10.8231, 106.6297),
    ("Cần Thơ", 10.0452, 105.7469),
    ("An Giang", 10.5216, 105.1259),
]

# =============================
# Sinh mock forecast data
# =============================
def generate_mock_forecast(days=5, seed=42):
    np.random.seed(seed)
    records = []

    base_date = date.today()
    for d in range(days):
        forecast_date = base_date + timedelta(days=d)
        for name, lat, lon in PROVINCES:
            records.append({
                "province": name,
                "lat": lat,
                "lon": lon,
                "date": forecast_date.isoformat(),
                "temp": np.random.uniform(18, 35),       # °C
                "humidity": np.random.uniform(50, 90),   # %
                "wind": np.random.uniform(0.2, 5.0),     # m/s
            })
    return pd.DataFrame(records)

df_forecast = generate_mock_forecast(days=7)

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Bản đồ dự báo thời tiết", layout="wide")

st.title("🌤️ Bản đồ dự báo thời tiết (Mock data)")
st.caption("Mô phỏng giao diện bản đồ dự báo theo tỉnh – chưa dùng model")

# Chọn ngày dự báo
selected_date = st.date_input(
    "📅 Chọn ngày dự báo",
    value=pd.to_datetime(df_forecast["date"].iloc[0])
)

df_show = df_forecast[df_forecast["date"] == selected_date.isoformat()]

# =============================
# Vẽ bản đồ
# =============================
fig = px.scatter_mapbox(
    df_show,
    lat="lat",
    lon="lon",
    size="temp",
    color="temp",
    color_continuous_scale="Reds",
    size_max=30,
    zoom=4.5,
    hover_name="province",
    hover_data={
        "date": True,
        "temp": ':.1f',
        "humidity": ':.1f',
        "wind": ':.2f',
        "lat": False,
        "lon": False
    },
    labels={
        "temp": "Nhiệt độ (°C)",
        "humidity": "Độ ẩm (%)",
        "wind": "Tốc độ gió (m/s)"
    },
    height=650
)

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":0,"l":0,"b":0}
)

st.plotly_chart(fig, use_container_width=True)

# =============================
# Bảng dữ liệu chi tiết
# =============================
with st.expander("📋 Xem bảng dự báo chi tiết"):
    st.dataframe(
        df_show.sort_values("temp", ascending=False),
        use_container_width=True
    )

st.info(
    "🔎 Đây là mock data để minh họa giao diện. "
    "Khi dùng thật, chỉ cần thay df_forecast bằng output từ model LSTM/RNN."
)
