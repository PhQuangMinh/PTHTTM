import json
import os
from datetime import datetime, timedelta, time

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "full.csv")
GEOJSON_PATH = os.path.join(BASE_DIR, "vn.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pm25_lstm.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "pm25_scaler.pkl")
WINDOW_SIZE = 24


@st.cache_data(show_spinner=False)
def load_pm25_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(
        csv_path,
        usecols=["parameter", "value", "datetimeLocal"],
        parse_dates=["datetimeLocal"],
        encoding="latin1",
        on_bad_lines="skip",
    )
    pm25 = (
        df[df["parameter"].str.lower() == "pm25"]
        .dropna(subset=["value", "datetimeLocal"])
        .assign(value=lambda d: d["value"].astype(float))
        .set_index("datetimeLocal")
        .sort_index()
        .resample("1H")
        .mean(numeric_only=True)
        .interpolate()
    )
    pm25 = pm25["value"]
    pm25.index = pm25.index.tz_localize(None)
    return pm25


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def stations_from_geojson(geojson: dict) -> pd.DataFrame:
    features = geojson.get("features") or []
    rows = []
    for feature in features:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) >= 2:
            rows.append(
                {
                    "name": props.get("name"),
                    "description": props.get("description"),
                    "lat": coords[1],
                    "lon": coords[0],
                }
            )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    # Lazy import để app render phần map nhanh hơn (TF import khá nặng)
    import tensorflow as tf

    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            return tf.keras.models.load_model(model_path)
    return None


@st.cache_data(show_spinner=False)
def load_scaler(path: str):
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_station_latest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=[
            "location_name",
            "parameter",
            "value",
            "datetimeLocal",
            "latitude",
            "longitude",
        ],
        parse_dates=["datetimeLocal"],
        encoding="latin1",
        on_bad_lines="skip",
    )
    # Sửa lỗi mã hóa tên trạm (latin1/cp1258 bị vỡ)
    def _fix_name(s: str) -> str:
        if not isinstance(s, str):
            return s
        candidates = [s]
        try:
            candidates.append(s.encode("latin1").decode("utf-8"))
        except Exception:
            pass
        try:
            candidates.append(s.encode("latin1").decode("cp1258"))
        except Exception:
            pass
        # Chọn chuỗi dài nhất (ít ký tự lỗi nhất)
        best = max(candidates, key=len)
        name_fixes = {
            "556 Nguy?n V?n C?": "556 Nguyễn Văn Cừ",
            "556 Nguyên Văn Cừ": "556 Nguyễn Văn Cừ",
            "Công viên h? ?i?u hòa Nhân Chính, Khu?t Duy Ti?n": "Công viên hồ điều hòa Nhân Chính, Khuất Duy Tiến",
            "Công viên hồ điều hòa Nhân Chính, Khu?t Duy Ti?n": "Công viên hồ điều hòa Nhân Chính, Khuất Duy Tiến",
            "?H Bách Khoa - c?ng Parabol ???ng Gi?i Phóng css Copy code": "ĐH Bách Khoa - cổng Parabol đường Giải Phóng",
            "?H Bách Khoa - c?ng Parabol ???ng Gi?i Phóng css Copy code": "ĐH Bách Khoa - cổng Parabol đường Giải Phóng",
            "OceanPark": "Ocean Park",
            "S\u00f4\u0301 46, ph\u00f4\u0301 L\u01b0u Quang V\u0169": "Số 46, phố Lưu Quang Vũ",
        }
        return name_fixes.get(best, best)

    df["location_name"] = df["location_name"].apply(_fix_name)
    df = df[df["parameter"].str.lower() == "pm25"].dropna(
        subset=["value", "datetimeLocal", "latitude", "longitude"]
    )
    df = df.assign(
        value=lambda d: d["value"].astype(float),
        datetimeLocal=lambda d: d["datetimeLocal"].dt.tz_localize(None),
    )
    # Lấy bản ghi mới nhất cho từng trạm
    idx = df.groupby("location_name")["datetimeLocal"].idxmax()
    latest = df.loc[idx].rename(
        columns={
            "value": "pm25",
            "datetimeLocal": "last_time",
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    latest["last_time_str"] = latest["last_time"].dt.strftime("%Y-%m-%d %H:%M")
    return latest[["location_name", "pm25", "lat", "lon", "last_time_str"]]


def pm25_color_rgba(v: float) -> list[int]:
    # AQI-like mức độ: xanh (tốt) -> đỏ (rất xấu)
    try:
        val = float(v)
    except Exception:
        return [120, 120, 120, 160]
    if val <= 12:
        return [0, 153, 102, 200]  # xanh
    if val <= 35:
        return [255, 222, 51, 220]  # vàng
    if val <= 55:
        return [255, 153, 51, 230]  # cam
    if val <= 150:
        return [238, 66, 102, 240]  # đỏ
    return [128, 0, 128, 255]  # tím cảnh báo


def build_sequence(series: pd.Series, scaler, window: int) -> np.ndarray:
    history = series.dropna().iloc[-window:]
    if len(history) < window:
        pad_size = window - len(history)
        first = history.iloc[:1].values if not history.empty else np.array([0.0])
        pad_values = np.repeat(first, pad_size)
        history = pd.Series(np.concatenate([pad_values, history.values]))
    scaled = scaler.transform(history.values.reshape(-1, 1))
    return scaled.reshape(1, window, 1)


def iterative_forecast(model, scaler, window_sequence: np.ndarray, steps: int) -> np.ndarray:
    seq = window_sequence.copy()
    preds = []
    for _ in range(steps):
        scaled_pred = model.predict(seq, verbose=0)
        pred = scaler.inverse_transform(scaled_pred)[0, 0]
        preds.append(pred)
        next_scaled = scaler.transform(np.array([[pred]]))
        seq = np.concatenate([seq[:, 1:, :], next_scaled.reshape(1, 1, 1)], axis=1)
    return np.array(preds)


def make_forecast_dataframe(last_timestamp: datetime, values: np.ndarray) -> pd.DataFrame:
    timestamps = [last_timestamp + timedelta(hours=i + 1) for i in range(len(values))]
    return pd.DataFrame({"datetime": timestamps, "pm25": values})


def main():
    st.set_page_config(
        page_title="Case10 - Dự báo chất lượng không khí",
        layout="wide",
        page_icon="🌫️",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .css-1lcbmhc {
            background-color: #ffffff;
        }
        .css-1y4p8pa {
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Dự báo PM2.5 – Case 10")
    st.caption("LSTM dữ liệu OpenAQ, bản đồ trạm và dự báo theo giờ.")

    pm25 = load_pm25_series(DATA_PATH)
    last_ts = pm25.index.max()
    st.subheader("Tổng quan dữ liệu")
    st.markdown(
        f"- Dữ liệu PM2.5 từ {pm25.index.min().date()} đến {last_ts.date()}.\n"
        f"- Cập nhật mới nhất: {last_ts.strftime('%Y-%m-%d %H:%M')}"
    )

    geojson = load_geojson(GEOJSON_PATH)
    stations_df = stations_from_geojson(geojson)

    st.subheader("Bản đồ chất lượng không khí")
    station_latest = load_station_latest(DATA_PATH)
    if station_latest.empty or stations_df.empty:
        st.warning("Không thể đọc dữ liệu trạm hoặc dữ liệu PM2.5 mới nhất.")
    else:
        station_latest = station_latest.merge(
            stations_df[["name", "description"]],
            left_on="location_name",
            right_on="name",
            how="left",
        )
        station_latest["description"] = station_latest["description"].fillna("")
        station_latest["color"] = station_latest["pm25"].apply(pm25_color_rgba)
        center_lat = float(station_latest["lat"].mean())
        center_lon = float(station_latest["lon"].mean())
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=station_latest,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=1200,
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {
            "html": "<b>{location_name}</b><br/>PM2.5: {pm25} µg/m³<br/>Cập nhật: {last_time_str}<br/>{description}",
            "style": {"backgroundColor": "white", "color": "black"},
        }
        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon, zoom=10, pitch=0
            ),
            layers=[layer],
            tooltip=tooltip,
        )
        st.pydeck_chart(deck, use_container_width=True)
        with st.expander("Danh sách trạm và PM2.5 mới nhất"):
            st.dataframe(
                station_latest[
                    ["location_name", "pm25", "last_time_str", "description", "lat", "lon"]
                ].rename(
                    columns={
                        "location_name": "Trạm",
                        "pm25": "PM2.5 (µg/m³)",
                        "last_time_str": "Cập nhật",
                        "description": "Mô tả",
                    }
                )
            )

    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)

    if model is None or scaler is None:
        return

    st.subheader("Thiết lập dự báo")
    col1, col2 = st.columns(2)
    future_date = col1.date_input(
        "Chọn ngày dự báo",
        value=(last_ts + timedelta(days=1)).date(),
        min_value=last_ts.date(),
        max_value=(last_ts + timedelta(days=14)).date(),
    )
    future_hour = col2.slider("Chọn giờ", 0, 23, 8)
    target_datetime = datetime.combine(future_date, time(hour=future_hour))
    horizon_hours = max(1, int((target_datetime - last_ts).total_seconds() // 3600))
    horizon_hours = max(horizon_hours, 1)
    forecast_steps = st.slider("Số giờ cần dự báo", 6, 72, 24)
    total_steps = max(forecast_steps, horizon_hours)

    run_forecast = st.button("Chạy dự báo", type="primary", use_container_width=True)
    if not run_forecast:
        st.info("Nhấn **Chạy dự báo** để tính toán. (Giúp phần bản đồ hiển thị nhanh hơn)")
        return

    with st.spinner("Đang tạo dự báo..."):
        seq = build_sequence(pm25, scaler, WINDOW_SIZE)
        preds = iterative_forecast(model, scaler, seq, total_steps)
        forecast_df = make_forecast_dataframe(last_ts, preds)

    target_pred = None
    if horizon_hours <= len(preds):
        target_pred = preds[horizon_hours - 1]
    actual_value = pm25.get(target_datetime, None)

    st.subheader("Kết quả dự báo")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Thời điểm", target_datetime.strftime("%Y-%m-%d %H:%M"))
    col_b.metric(
        "Dự đoán PM2.5",
        f"{target_pred:.1f} µg/m³" if target_pred is not None else "Chưa có dữ liệu",
    )
    col_c.metric(
        "Quan sát thực tế",
        f"{actual_value:.1f} µg/m³" if actual_value is not None else "Không có",
    )

    st.line_chart(
        forecast_df.rename(columns={"datetime": "index"}).set_index("index")["pm25"].rename("Dự báo PM2.5"),
        use_container_width=True,
    )

    st.subheader("PM2.5 thực tế (72 giờ gần nhất)")
    st.line_chart(
        pm25.tail(72).rename("PM2.5 thực tế"),
        use_container_width=True,
    )

    st.markdown(
        "## Hướng dẫn triển khai\n"
        "- Đảm bảo thư mục `models` chứa `pm25_lstm.keras` và `pm25_scaler.pkl`.\n"
        "- Chạy `streamlit run app.py` trong `case10-chat-luong-kk` để mở dashboard local.\n"
        "- Cập nhật `vn.json` để bổ sung trạm mới, Streamlit sẽ tự động vẽ lại bản đồ."
    )


if __name__ == "__main__":
    main()
