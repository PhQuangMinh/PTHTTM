import os
import pickle
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st


def try_joblib_load(path: str):
	try:
		import joblib  # type: ignore
		return joblib.load(path)
	except Exception as e:
		print(f"[load] joblib.load failed for '{path}': {e}")
		return None


def try_pickle_load(path: str):
	try:
		with open(path, "rb") as f:
			return pickle.load(f)
	except Exception as e:
		print(f"[load] pickle.load failed for '{path}': {e}")
		return None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_TITLE = "Dự đoán nguy cơ bệnh tiểu đường"

MODEL_FILES = [
	("mlp7", "diabetes_mlp7.sav", "MLP-7"),
	("model", "diabetes_model.sav", "Model tổng hợp"),
]

# Thứ tự đặc trưng theo bộ Pima Indians Diabetes (8 đặc trưng)
FEATURES: List[Dict[str, Any]] = [
	{"name": "pregnancies", "label": "Số lần mang thai", "placeholder": "0", "step": "1", "type": "number", "default": "0"},
	{"name": "glucose", "label": "Glucose (mg/dL)", "placeholder": "120", "step": "0.1", "type": "number", "default": "120"},
	{"name": "blood_pressure", "label": "Huyết áp tâm trương (mm Hg)", "placeholder": "70", "step": "0.1", "type": "number", "default": "70"},
	{"name": "skin_thickness", "label": "Độ dày da (mm)", "placeholder": "20", "step": "0.1", "type": "number", "default": "20"},
	{"name": "insulin", "label": "Insulin (mu U/ml)", "placeholder": "80", "step": "0.1", "type": "number", "default": "80"},
	{"name": "bmi", "label": "BMI", "placeholder": "26.5", "step": "0.1", "type": "number", "default": "26.5"},
	{"name": "diabetes_pedigree", "label": "Diabetes Pedigree Function", "placeholder": "0.5", "step": "0.01", "type": "number", "default": "0.5"},
	{"name": "age", "label": "Tuổi", "placeholder": "33", "step": "1", "type": "number", "default": "33"},
]

# Khi mô hình chỉ yêu cầu 5 đặc trưng, ưu tiên 5 đặc trưng phổ biến
FEATURES_TOP5_NAMES = ["glucose", "bmi", "age", "insulin", "pregnancies"]


def get_expected_feature_count(model_obj: Any) -> Optional[int]:
	"""
	Cố gắng suy ra số đặc trưng đầu vào mà mô hình mong đợi.
	- Ưu tiên thuộc tính n_features_in_ của estimator hoặc transformer trong Pipeline
	- Trả về None nếu không thể suy ra
	"""
	# Trường hợp Pipeline
	try:
		from sklearn.pipeline import Pipeline  # type: ignore
		if isinstance(model_obj, Pipeline):
            # Thử tìm trên bước đầu vào (transformer) hoặc estimator cuối cùng
			for name, step in model_obj.named_steps.items():
				n_in = getattr(step, "n_features_in_", None)
				if isinstance(n_in, int) and n_in > 0:
					return n_in
			# fallback: estimator cuối cùng
			final_est = getattr(model_obj, "steps", [])[-1][1] if getattr(model_obj, "steps", []) else None
			n_in = getattr(final_est, "n_features_in_", None)
			if isinstance(n_in, int) and n_in > 0:
				return n_in
	except Exception:
		pass

	# Trường hợp estimator đơn lẻ có n_features_in_
	n_in = getattr(model_obj, "n_features_in_", None)
	if isinstance(n_in, int) and n_in > 0:
		return n_in

	return None


def select_feature_schema_for_model(model_obj: Any) -> List[Dict[str, Any]]:
	"""
	Chọn danh sách đặc trưng hiển thị/phân tích theo số đặc trưng mô hình mong đợi.
	- 8: dùng đủ FEATURES
	- 5: dùng 5 đặc trưng phổ biến trong Pima
	- Mặc định: dùng đủ FEATURES
	"""
	expected = get_expected_feature_count(model_obj)
	if expected == 5:
		# Lấy theo thứ tự ưu tiên trong FEATURES_TOP5_NAMES
		name_to_feature = {f["name"]: f for f in FEATURES}
		selected = [name_to_feature[n] for n in FEATURES_TOP5_NAMES if n in name_to_feature]
		# nếu thiếu vì khác tên, fallback sang 5 đầu tiên
		if len(selected) == 5:
			return selected
		return FEATURES[:5]
	# Nếu là 8 (chuẩn Pima) hoặc không xác định: dùng đủ 8
	return FEATURES
 

def load_available_models() -> Dict[str, Dict[str, Any]]:
	models: Dict[str, Dict[str, Any]] = {}
	for key, filename, display_name in MODEL_FILES:
		path = os.path.join(BASE_DIR, filename)
		print(path)
		if not os.path.exists(path):
			print(f"[load] file not found, skip: {path}")
			continue
		model_obj = try_joblib_load(path) or try_pickle_load(path)
		if model_obj is not None:
			features_schema = select_feature_schema_for_model(model_obj)
			expected_n = get_expected_feature_count(model_obj)
			models[key] = {
				"model": model_obj,
				"name": display_name,
				"file": filename,
				"features": features_schema,
				"expected_n": expected_n,
			}
		else:
			print(f"[load] unable to deserialize model from '{path}' using joblib/pickle.")
	return models


def parse_input_values(form_data: Dict[str, str], features_schema: List[Dict[str, Any]]) -> Tuple[Optional[List[float]], Optional[str], Dict[str, str]]:
	values: List[float] = []
	kept: Dict[str, str] = {}
	for feature in features_schema:
		name = feature["name"]
		raw = (form_data.get(name) or "").strip()
		kept[name] = raw
		if raw == "":
			return None, f"Vui lòng nhập '{feature['label']}'.", kept
		try:
			values.append(float(raw))
		except ValueError:
			return None, f"Giá trị không hợp lệ cho '{feature['label']}'.", kept
	return values, None, kept


def predict_with_model(model: Any, x: np.ndarray) -> Tuple[int, Optional[float]]:
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(x)
		if isinstance(proba, list):
			proba = np.array(proba)
		if proba.ndim == 2 and proba.shape[1] >= 2:
			p1 = float(proba[0, 1])
			label = 1 if p1 >= 0.5 else 0
			return label, p1
	# Fallback predict
	y_pred = model.predict(x)
	if isinstance(y_pred, list):
		y_pred = np.array(y_pred)
	if y_pred.ndim == 1:
		label = int(round(float(y_pred[0])))
	else:
		label = int(round(float(y_pred.ravel()[0])))
	return label, None


def main():
	st.set_page_config(page_title=APP_TITLE, layout="centered")

	# CSS trắng/đen, in A4 rõ nét
	st.markdown(
		"""
		<style>
		:root { color-scheme: light; }
		html, body, [data-testid="stApp"] { background: #ffffff !important; color: #000000 !important; }
		.block-container { max-width: 1000px; }
		h1, h2, h3, h4, h5, h6, p, label, span, div { color: #000000 !important; }
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
	st.write("Nhập đầy đủ thông số Pima, chọn mô hình và dự đoán nguy cơ tiểu đường.")

	@st.cache_resource(show_spinner=False)
	def cached_models() -> Dict[str, Dict[str, Any]]:
		return load_available_models()

	models = cached_models()
	if not models:
		st.error("Không tìm thấy hoặc không nạp được mô hình .sav trong thư mục. Hãy cài đặt scikit-learn/joblib đúng phiên bản.")
		st.stop()

	keys = list(models.keys())
	default_key = "mlp7" if "mlp7" in keys else keys[0]

	col_left, col_right = st.columns([1.2, 0.8])

	with col_right:
		st.subheader("Mô hình")
		model_key = st.radio(
			"Chọn mô hình",
			options=keys,
			index=keys.index(default_key),
			format_func=lambda k: f"{models[k]['name']} ({models[k]['file']})",
		)

	with col_left:
		st.subheader("Thông số đầu vào")
		values: Dict[str, float] = {}

		# Khoảng giá trị hợp lý cho từng đặc trưng
		FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
			"pregnancies": (0.0, 20.0),
			"glucose": (50.0, 300.0),
			"blood_pressure": (40.0, 200.0),
			"skin_thickness": (0.0, 100.0),
			"insulin": (0.0, 900.0),
			"bmi": (10.0, 70.0),
			"diabetes_pedigree": (0.0, 3.0),
			"age": (10.0, 100.0),
		}

		icon_map = {
			"pregnancies": "🤰",
			"glucose": "🩸",
			"blood_pressure": "💓",
			"skin_thickness": "🧪",
			"insulin": "💉",
			"bmi": "⚖️",
			"diabetes_pedigree": "🧬",
			"age": "🎂",
		}

		# Hiển thị inputs theo lưới
		grid_cols = st.columns(2)
		for idx, f in enumerate(FEATURES):
			col = grid_cols[idx % 2]
			with col:
				placeholder = f.get("placeholder", "")
				default_val = float(f.get("default", "0"))
				rmin, rmax = FEATURE_RANGES.get(f["name"], (0.0, 1_000_000.0))
				# Bắt buộc số nguyên
				rmin_i = int(rmin)
				rmax_i = int(rmax)
				default_i = int(round(default_val))
				# Đảm bảo giá trị mặc định nằm trong khoảng
				if default_i < rmin_i:
					default_i = rmin_i
				elif default_i > rmax_i:
					default_i = rmax_i
				label_icon = f"{icon_map.get(f['name'], '•')} {f['label']} (khoảng: {rmin_i}–{rmax_i})"
				help_text = "Nhập số nguyên trong khoảng trên."
				values[f["name"]] = st.number_input(
					label_icon,
					value=default_i,
					help=help_text,
					step=1,
					min_value=rmin_i,
					max_value=rmax_i,
				)

		# Xác định đặc trưng thực sự sẽ dùng theo mô hình
		expected_n = models[model_key].get("expected_n") or 8
		if expected_n == 5:
			used_feature_names = FEATURES_TOP5_NAMES
			st.caption("Mô hình sẽ sử dụng 5 đặc trưng: Glucose, BMI, Tuổi, Insulin, Số lần mang thai.")
		else:
			used_feature_names = [f["name"] for f in FEATURES]
			st.caption("Mô hình sẽ sử dụng đầy đủ 8 đặc trưng Pima.")

	if st.button("Dự đoán", type="primary"):
		try:
			x = np.array([values[name] for name in used_feature_names], dtype=float).reshape(1, -1)
			label, prob = predict_with_model(models[model_key]["model"], x)

			st.subheader("Kết quả")
			lbl = "⚠️ Có nguy cơ" if label == 1 else "✅ Ít nguy cơ"
			st.markdown(f'<span class="badge">{lbl}</span>', unsafe_allow_html=True)
			if prob is not None:
				st.write(f"Xác suất nguy cơ: {prob*100:.2f}%")
				st.markdown(f'<div class="pb"><div style="width:{prob*100:.2f}%"></div></div>', unsafe_allow_html=True)
		except Exception as e:
			st.error(f"Lỗi khi dự đoán: {e}")

	# Miêu tả dưới app
	st.divider()
	st.subheader("Miêu tả tham số (theo bộ dữ liệu Pima)")
	st.markdown(
		"""
		- 🤰 Số lần mang thai: tổng số lần mang thai của bệnh nhân.
		- 🩸 Glucose: nồng độ glucose huyết tương sau 2 giờ (mg/dL).
		- 💓 Huyết áp tâm trương: mm Hg.
		- 🧪 Độ dày da (Skin Thickness): mm.
		- 💉 Insulin: nồng độ insulin 2 giờ (mu U/ml).
		- ⚖️ BMI: chỉ số khối cơ thể.
		- 🧬 Diabetes Pedigree Function: chỉ số di truyền liên quan tiểu đường.
		- 🎂 Tuổi: tuổi (năm).
		"""
	)

if __name__ == "__main__":
	main()


