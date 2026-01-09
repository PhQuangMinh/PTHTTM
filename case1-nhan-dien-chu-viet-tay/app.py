import io
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="MNIST CNN Demo",
    page_icon="🔢",
    layout="centered",
)
# Force light theme: white background, black text across the app
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stHeader"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    body, [class*="css"] {
        color: #000000 !important;
        background: #ffffff !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Paths & constants =====
MODEL_PATH = pathlib.Path("keras_cnn.keras")
TEST_CSV = pathlib.Path("data/mnist_test.csv")
IMG_SHAPE = (28, 28)
NUM_CHANNELS = 1


@st.cache_resource(show_spinner="Đang tải mô hình Keras ...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file model tại {MODEL_PATH.resolve()}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner="Đang tải test set ...")
def load_test_data(limit: int | None = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc mnist_test.csv (cột 0=label, còn lại=pixel). Lấy mẫu nhỏ để đánh giá nhanh."""
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy test csv tại {TEST_CSV.resolve()}")
    df = pd.read_csv(TEST_CSV)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=42)
    labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
    images = df.iloc[:, 1:].to_numpy(dtype=np.float32).reshape(-1, 28, 28, 1) / 255.0
    return images, labels


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Chuyển ảnh bất kỳ sang tensor 28x28x1 (float32, 0-1)."""
    img = img.convert("L").resize(IMG_SHAPE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def evaluate_model(model) -> float:
    x_test, y_test = load_test_data()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return float(acc)


# ===== UI =====
st.title("MNIST CNN Demo")
st.caption("Nền trắng, chữ đen. Mô tả: phân loại chữ số viết tay bằng CNN (Keras) và cho phép xem độ chính xác nhanh.")



model_status = "✅ Đã tìm thấy model" if MODEL_PATH.exists() else "⚠️ Chưa có file model"
st.write(model_status, "-", MODEL_PATH)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Độ chính xác")
    if st.button("Tính accuracy trên test set (mẫu)", type="primary"):
        try:
            model = load_model()
            acc = evaluate_model(model)
            st.success(f"Accuracy (mẫu test): {acc:.4f}")
        except Exception as e:
            st.error(f"Lỗi khi đánh giá: {e}")

with col2:
    st.subheader("Dự đoán ảnh tải lên")
    file = st.file_uploader("Tải ảnh (png/jpg/bmp). Ảnh sẽ được chuyển về 28x28)", type=["png", "jpg", "jpeg", "bmp"])
    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            st.image(image, caption="Ảnh gốc", width=150)
            x = preprocess_image(image)
            model = load_model()
            probs = model.predict(x, verbose=0)[0]
            pred = int(np.argmax(probs))
            st.success(f"Kết quả dự đoán: {pred}")
            st.bar_chart(probs)
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

st.divider()

st.subheader("Ghi chú")
st.markdown(
    """
- Accuracy dùng mẫu test 2,000 ảnh để nhanh; chỉnh `limit` trong `load_test_data` nếu muốn full.
- Để chạy: `streamlit run app.py`
- Nếu thiếu thư viện: `pip install streamlit tensorflow pandas pillow`
"""
)
