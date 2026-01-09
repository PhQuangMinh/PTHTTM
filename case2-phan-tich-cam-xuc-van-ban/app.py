import os
import pickle
from typing import Dict, Tuple, Optional, List

import numpy as np
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
	# Prefer importing tensorflow as a whole to keep compatibility across TF versions
	import tensorflow as tf  # noqa: F401
	from tensorflow import keras
	from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as import_error:  # pragma: no cover
	raise RuntimeError(
		f"Không thể import TensorFlow/Keras. Vui lòng cài đặt TensorFlow trước. Chi tiết: {import_error}"
	)


APP_TITLE = "Nhận diện cảm xúc văn bản (LSTM / GRU)"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.keras")
GRU_MODEL_PATH = os.path.join(BASE_DIR, "gru_model.keras")


@st.cache_resource(show_spinner=False)
def load_tokenizer(tokenizer_path: str):
	with open(tokenizer_path, "rb") as f:
		tokenizer = pickle.load(f)
	return tokenizer


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, keras.Model]:
	models: Dict[str, keras.Model] = {}

	if os.path.exists(LSTM_MODEL_PATH):
		models["lstm"] = keras.models.load_model(LSTM_MODEL_PATH, compile=False)
	if os.path.exists(GRU_MODEL_PATH):
		models["gru"] = keras.models.load_model(GRU_MODEL_PATH, compile=False)

	if not models:
		raise FileNotFoundError(
			"Không tìm thấy file mô hình. Cần có ít nhất một trong các file: "
			f"'{os.path.basename(LSTM_MODEL_PATH)}' hoặc '{os.path.basename(GRU_MODEL_PATH)}'."
		)

	return models


def get_sequence_length_from_model(model: keras.Model) -> Optional[int]:
	"""
	Cố gắng suy ra độ dài chuỗi (max_len) từ input_shape của mô hình.
	Trả về None nếu không suy ra được.
	"""
	input_shape = model.input_shape

	# Trường hợp nhiều đầu vào: lấy đầu vào đầu tiên có dạng (None, seq_len)
	if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
		for shape in input_shape:
			if isinstance(shape, (list, tuple)) and len(shape) >= 2 and shape[0] is None and isinstance(shape[1], int):
				return shape[1]

	# Trường hợp 1 đầu vào
	if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
		# Kỳ vọng dạng (None, seq_len) hoặc (None, seq_len, features)
		seq_dim = input_shape[1]
		if isinstance(seq_dim, int):
			return seq_dim

	return None


def preprocess_texts(
	texts: List[str],
	tokenizer,
	max_len: int,
) -> np.ndarray:
	sequences = tokenizer.texts_to_sequences(texts)
	padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
	return padded


def interpret_prediction(pred: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
	"""
	Trả về (nhãn_dự_đoán, độ_tin_cậy, danh_sách_nhãn_và_xác_suất).
	Hỗ trợ:
	- Nhị phân sigmoid: shape (1, 1)
	- Softmax nhiều lớp: shape (1, C)
	"""
	if pred.ndim == 2 and pred.shape[0] == 1:
		if pred.shape[1] == 1:
			# Nhị phân
			score = float(pred[0, 0])
			label = "Tích cực" if score >= 0.5 else "Tiêu cực"
			confidence = score if score >= 0.5 else 1.0 - score
			distribution = [("Tiêu cực", 1.0 - score), ("Tích cực", score)]
			return label, float(confidence), distribution

		# Nhiều lớp
		probs = pred[0].astype(float)
		class_idx = int(np.argmax(probs))
		num_classes = pred.shape[1]

		# Gợi ý nhãn phổ biến
		default_labels = {
			2: ["Tiêu cực", "Tích cực"],
			3: ["Tiêu cực", "Trung lập", "Tích cực"],
		}
		labels = default_labels.get(num_classes, [f"Lớp {i}" for i in range(num_classes)])

		top_label = labels[class_idx] if class_idx < len(labels) else f"Lớp {class_idx}"
		confidence = float(probs[class_idx])
		distribution = [(labels[i] if i < len(labels) else f"Lớp {i}", float(p)) for i, p in enumerate(probs)]
		return top_label, confidence, distribution

	# Phòng thủ
	return "Không xác định", 0.0, []


def main():
	st.set_page_config(page_title=APP_TITLE, layout="centered")

	# CSS nền trắng, chữ đen, in ấn rõ ràng
	st.markdown(
		"""
		<style>
		:root { color-scheme: light; }
		html, body, [data-testid="stApp"] { background: #ffffff !important; color: #000000 !important; }
		.block-container { max-width: 900px; }
		h1, h2, h3, h4, h5, h6, p, label, span, div { color: #000000 !important; }
		.section-hint { color: #111; font-size: 14px; opacity: .85; }
		.badge { display: inline-block; padding: 4px 10px; border: 1px solid #000; border-radius: 999px; font-size: 12px; font-weight: 700; color:#000; background: transparent; }
		.pb { height: 10px; background: #e5e7eb; border-radius: 6px; overflow: hidden; }
		.pb > div { height: 100%; background: #111; }
		@media print {
			@page { size: A4; margin: 12mm; }
			* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
			html, body, [data-testid="stApp"] { background: #ffffff !important; color: #000000 !important; }
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.title(APP_TITLE)
	st.write("Nhập văn bản tiếng Việt, chọn mô hình LSTM/GRU và xem kết quả dự đoán cảm xúc.")

	with st.spinner("Đang tải mô hình/tokenizer..."):
		tokenizer = load_tokenizer(TOKENIZER_PATH)
		models = load_models()

	available = list(models.keys())
	default_model = "lstm" if "lstm" in available else ("gru" if "gru" in available else available[0])

	col1, col2 = st.columns([1.2, 0.8])
	with col1:
		st.subheader("Văn bản đầu vào")
		text = st.text_area(
			"Nhập văn bản",
			"Ví dụ: It is so beautiful!",
			height=160,
			label_visibility="collapsed",
		)
		st.caption("Văn bản sẽ được xử lý dựa trên tokenizer đã được huấn luyện.")
	with col2:
		st.subheader("Mô hình")
		model_name = st.radio("Chọn mô hình", options=available, index=available.index(default_model))
	
	predict_clicked = st.button("Dự đoán cảm xúc", type="primary")

	if predict_clicked:
		if not text.strip():
			st.error("Vui lòng nhập văn bản.")
			return

		model = models[model_name]
		seq_len = get_sequence_length_from_model(model) or 200

		try:
			padded = preprocess_texts([text.strip()], tokenizer, seq_len)
			raw_pred = model.predict(padded, verbose=0)
			label, confidence, distribution = interpret_prediction(np.array(raw_pred))

			st.subheader("Kết quả")
			badge = f'<span class="badge">{label}</span>'
			st.markdown(badge, unsafe_allow_html=True)
			st.write(f"Độ tin cậy: {confidence*100:.2f}%")

			if distribution:
				st.write("Phân bố xác suất:")
				for name, prob in distribution:
					st.write(f"- {name}: {prob*100:.1f}%")
					# progress bar đen trắng
					st.markdown(f'<div class="pb"><div style="width:{prob*100:.2f}%"></div></div>', unsafe_allow_html=True)
		except Exception as e:
			st.error(f"Lỗi khi dự đoán: {e}")


if __name__ == "__main__":
	main()


