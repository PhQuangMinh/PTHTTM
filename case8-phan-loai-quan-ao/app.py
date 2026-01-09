import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================
# UI + Theme (white bg, black text)
# ============================
st.set_page_config(
    page_title="Nhận diện quần áo",
    page_icon="👕",
    layout="centered",
)

st.markdown(
    """
<style>
/* Force white background + black text everywhere */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
  background-color: #ffffff !important;
  color: #000000 !important;
}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
  background-color: #ffffff !important;
  color: #000000 !important;
}

/* Inputs & widgets */
input, textarea, select, button, [role="textbox"], [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] {
  background-color: #ffffff !important;
  color: #000000 !important;
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
  background-color: #ffffff !important;
  border: 1px solid #00000022 !important;
}

/* Keep charts readable */
[data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"], svg {
  background-color: #ffffff !important;
}

/* Metric cards / containers */
[data-testid="stMetric"], [data-testid="stMetric"] * {
  color: #000000 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================
# Labels (from notebook)
# ============================
CLASS_NAMES = [
    "Blazer",
    "Celana_Panjang",
    "Celana_Pendek",
    "Gaun",
    "Hoodie",
    "Jaket",
    "Jaket_Denim",
    "Jaket_Olahraga",
    "Jeans",
    "Kaos",
    "Kemeja",
    "Mantel",
    "Polo",
    "Rok",
    "Sweter",
]

VI_LABELS = {
    "Blazer": "Áo blazer",
    "Celana_Panjang": "Quần dài",
    "Celana_Pendek": "Quần short",
    "Gaun": "Váy/đầm",
    "Hoodie": "Áo hoodie",
    "Jaket": "Áo khoác",
    "Jaket_Denim": "Áo khoác denim",
    "Jaket_Olahraga": "Áo khoác thể thao",
    "Jeans": "Quần jeans",
    "Kaos": "Áo thun",
    "Kemeja": "Áo sơ mi",
    "Mantel": "Áo măng tô",
    "Polo": "Áo polo",
    "Rok": "Chân váy",
    "Sweter": "Áo len",
}


def pretty_label(class_name: str) -> str:
    vi = VI_LABELS.get(class_name, class_name)
    return f"{vi} ({class_name})"


# ============================
# Model loading + inference
# ============================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def infer_input_spec(model):
    """Return (h, w, c, channels_first)."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"Unsupported model input_shape: {shape}")

    # Common: (None, H, W, C)
    if len(shape) == 4:
        # Try detect channels_last vs channels_first
        if shape[-1] in (1, 3):
            h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
            return h, w, c, False
        if shape[1] in (1, 3):
            c, h, w = int(shape[1]), int(shape[2]), int(shape[3])
            return h, w, c, True

        # Fallback assume channels_last
        h = int(shape[1]) if shape[1] is not None else 224
        w = int(shape[2]) if shape[2] is not None else 224
        c = int(shape[3]) if shape[3] is not None else 3
        return h, w, c, False

    # Sometimes: (None, H, W)
    if len(shape) == 3:
        h = int(shape[1]) if shape[1] is not None else 224
        w = int(shape[2]) if shape[2] is not None else 224
        return h, w, 1, False

    raise ValueError(f"Unsupported model input_shape length: {shape}")


def preprocess_pil(img_pil, h: int, w: int, c: int, channels_first: bool):
    from PIL import Image

    if c == 1:
        img_pil = img_pil.convert("L")
    else:
        img_pil = img_pil.convert("RGB")

    img_pil = img_pil.resize((w, h), Image.BILINEAR)
    x = np.asarray(img_pil).astype(np.float32)

    if c == 1:
        if x.ndim == 2:
            x = x[..., None]
        elif x.ndim == 3 and x.shape[-1] != 1:
            x = x.mean(axis=-1, keepdims=True)

    # Normalize to [0,1]
    if x.max() > 1.0:
        x = x / 255.0

    # Add batch dim
    x = np.expand_dims(x, axis=0)

    if channels_first:
        # (1, H, W, C) -> (1, C, H, W)
        x = np.transpose(x, (0, 3, 1, 2))

    return x


def predict(model, x):
    probs = model.predict(x, verbose=0)
    probs = np.asarray(probs)

    if probs.ndim == 2:
        probs = probs[0]

    # If logits, softmax
    if probs.min() < 0 or probs.max() > 1.0 or not np.isclose(probs.sum(), 1.0, atol=1e-2):
        e = np.exp(probs - np.max(probs))
        probs = e / (e.sum() + 1e-12)

    return probs


# ============================
# App UI
# ============================
st.markdown(
    """
<div style="display:flex; align-items:center; gap:12px; margin-bottom: 4px;">
  <div style="font-size: 40px; line-height: 1;">👕</div>
  <div>
    <div style="font-size: 30px; font-weight: 800;">Nhận diện quần áo</div>
    <div style="font-size: 14px; opacity: 0.8;">Tải ảnh hoặc chụp ảnh để phân loại vào 15 nhóm quần áo.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Tuỳ chọn")
    show_topk = st.slider("Hiển thị Top-K", min_value=1, max_value=10, value=5, step=1)
    st.markdown("---")
    st.markdown(
        """
**Gợi ý ảnh tốt**
- 1 món đồ rõ ràng, ít nền rối
- Đủ sáng, không bị mờ
"""
    )

base_dir = Path(__file__).resolve().parent
model_path = str(base_dir / "clothes_cnn.keras")

col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    st.markdown("### Chọn ảnh")
    tab_up, tab_cam = st.tabs(["📁 Tải ảnh", "📷 Camera"])

    with tab_up:
        uploaded = st.file_uploader(
            "Chọn ảnh quần áo (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        img_bytes_upload = uploaded.getvalue() if uploaded else None

    with tab_cam:
        cam = st.camera_input("Chụp ảnh")
        img_bytes_cam = cam.getvalue() if cam else None

    # Prefer uploaded image if present, otherwise fallback to camera
    img_bytes = img_bytes_upload or img_bytes_cam

    if img_bytes:
        from PIL import Image

        img_pil = Image.open(io.BytesIO(img_bytes))
        st.image(img_pil, caption="Ảnh đầu vào", use_container_width=True)

with col_right:
    st.markdown("### Kết quả")

    if not os.path.exists(model_path):
        st.error(
            "Không tìm thấy model `clothes_cnn.keras` trong thư mục `case8-phan-loai-quan-ao`. "
            "Hãy đảm bảo file model nằm đúng vị trí."
        )
        st.stop()

    try:
        model = load_model(model_path)
        h, w, c, channels_first = infer_input_spec(model)
    except Exception as e:
        st.error(f"Không load được model. Lỗi: {e}")
        st.stop()

    st.caption(f"Input model: {h}×{w}×{c} ({'channels_first' if channels_first else 'channels_last'})")

    # Reset cached prediction if user changes image
    if img_bytes:
        image_id = hash(img_bytes)
        if st.session_state.get("last_image_id") != image_id:
            st.session_state["last_image_id"] = image_id
            st.session_state["last_probs"] = None

    run_predict = st.button("🔎 Dự đoán", type="primary", disabled=not bool(img_bytes))

    if not img_bytes:
        st.info("Hãy tải ảnh hoặc chụp ảnh, sau đó bấm **Dự đoán**.")
        st.stop()

    from PIL import Image

    img_pil = Image.open(io.BytesIO(img_bytes))
    x = preprocess_pil(img_pil, h=h, w=w, c=c, channels_first=channels_first)

    if run_predict:
        with st.spinner("Đang dự đoán..."):
            st.session_state["last_probs"] = predict(model, x)

    probs = st.session_state.get("last_probs")
    if probs is None:
        st.info("Bấm **Dự đoán** để xem kết quả.")
        st.stop()

    num_classes = int(probs.shape[-1])
    if num_classes != len(CLASS_NAMES):
        # Fallback if model classes mismatch
        class_names = [f"Class_{i}" for i in range(num_classes)]
        labels = class_names
    else:
        labels = CLASS_NAMES

    topk = int(min(show_topk, len(labels)))
    top_idx = np.argsort(-probs)[:topk]

    best_i = int(top_idx[0])
    best_label = labels[best_i]
    best_score = float(probs[best_i])

    st.markdown("#### Dự đoán")
    st.metric(label="Kết quả", value=pretty_label(best_label), delta=f"{best_score*100:.2f}%")

    st.markdown("#### Xác suất (Top-K)")
    df = pd.DataFrame(
        {
            "Lớp": [pretty_label(labels[i]) for i in top_idx],
            "Xác suất": [float(probs[i]) for i in top_idx],
        }
    )

    st.bar_chart(df.set_index("Lớp"), use_container_width=True)

    with st.expander("Xem chi tiết xác suất"):
        st.dataframe(
            df.assign(**{"Xác suất (%)": (df["Xác suất"] * 100).round(2)}).drop(columns=["Xác suất"]),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.caption("Nền trắng + chữ đen được ép bằng CSS để đúng yêu cầu.")



