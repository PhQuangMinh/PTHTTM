import os
import re
import pickle
from typing import List

import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Cấu hình port mong muốn (8506) nếu chạy bằng `streamlit run`
os.environ.setdefault("STREAMLIT_SERVER_PORT", "8506")

# -----------------------------
# Đường dẫn và cấu hình
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer_spam.pkl")
MODEL_FILES = {
    "LSTM": os.path.join(BASE_DIR, "spam_lstm_best.keras"),
    "Dense": os.path.join(BASE_DIR, "spam_dense_best.keras"),
}
MAX_LEN = 100

# -----------------------------
# Tiền xử lý văn bản
# -----------------------------


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_FILES.items():
        models[name] = load_model(path)
    return models


def predict(model_name: str, texts: List[str]) -> List[float]:
    tokenizer = load_tokenizer()
    models = load_models()
    model = models[model_name]

    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0).flatten()
    return probs.tolist()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Phân loại tin nhắn spam", page_icon="📩", layout="wide")

st.markdown(
    """
    <style>
    .main, .stApp { background: #FFFFFF !important; color: #111111 !important; }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { background: #FFFFFF !important; color: #111111 !important; }
    .stMetric { background: #fff; color: #111111; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #111111 !important; }
    .stSelectbox div, .stSlider, .stSlider * { color: #111111 !important; }
    input, textarea, select, option {
        color: #111111 !important;
        background: #FFFFFF !important;
        border: 1px solid #111111 !important;
    }

    /* ===== THÊM Ở ĐÂY: style cho st.code ===== */
    pre {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1.5px solid #111111 !important;
        border-radius: 6px !important;
    }

    pre code {
        color: #111111 !important;
        background-color: #FFFFFF !important;
        font-size: 14px;
        line-height: 1.6;
    }
    /* ===== Button trắng – chữ đen – viền đen ===== */
div.stButton > button {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    border: 1.5px solid #111111 !important;
    border-radius: 6px !important;
    font-weight: 500;
}

/* Hover */
div.stButton > button:hover {
    background-color: #F5F5F5 !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
}

/* Click */
div.stButton > button:active {
    background-color: #EDEDED !important;
    border-color: #000000 !important;
}

/* Focus (bỏ viền xanh mặc định) */
div.stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📩 Phân loại tin nhắn Spam / Ham")
st.caption("Nhập tin nhắn và xem kết quả của cả 2 mô hình (LSTM & Dense). Nền trắng chữ đen để dễ in báo cáo.")
THRESHOLD = 0.5  # ngưỡng cố định (spam nếu >= 0.5)

st.subheader("Nhập tin nhắn")
default_text = "Congratulations! You have won a $1,000 gift card. Click the link now to claim your prize!"
input_text = st.text_area("Nội dung tin nhắn", value=default_text, height=160)

col_run, col_clear = st.columns([1, 1])
with col_run:
    run = st.button("Phân loại", type="primary")
with col_clear:
    if st.button("Xóa nội dung"):
        input_text = ""

if run and input_text.strip():
    text_raw = input_text.strip()
    cleaned_text = clean_text(text_raw)

    results = []
    for model_name in MODEL_FILES.keys():
        prob = predict(model_name, [text_raw])[0]
        label = "Spam" if prob >= THRESHOLD else "Không spam"
        results.append((model_name, prob, label))

    cols = st.columns(len(results))
    for col, (model_name, prob, label) in zip(cols, results):
        with col:
            st.metric(f"{model_name} - Xác suất Spam", f"{prob*100:.1f}%")
            st.metric(f"{model_name} - Kết luận", label)

else:
    st.info("Nhập tin nhắn và nhấn 'Phân loại' để xem kết quả.")
# Hi John, don’t forget about our meeting tomorrow at 10 a.m. Let me know if you need anything.