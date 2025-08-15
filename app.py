import os
import io
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# Paths / constants
# ----------------------------
DATA_DIR = Path("data")
PICKLE_PATH = Path("data.pickle")
LABELMAP_PATH = Path("label_map.json")
MODEL_PATH = Path("model.p")

# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_image_any(input_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def list_labels() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])

def default_label_map_from_folders() -> dict:
    # Stable alphabetical mapping -> int ids
    labels = list_labels()
    return {lab: i for i, lab in enumerate(labels)}

# ----------------------------
# MediaPipe setup (reused)
# ----------------------------
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

@st.cache_resource
def mp_hands_detector(static=False):
    return mp_hands.Hands(
        static_image_mode=static,
        max_num_hands=2,
        model_complexity=1,  # Explicit in new versions
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3 if not static else 0.3
    )

def extract_features_from_bgr(bgr_img: np.ndarray) -> list[list[float]]:
    """
    Returns a list of feature vectors (one per detected hand),
    where each vector is [x1-min(x), y1-min(y), x2-min(x), y2-min(y), ...].
    Matches your original preprocessing (translation-invariant, not scale-invariant).
    """
    rgb = bgr_to_rgb(bgr_img)
    results = mp_hands_detector(static=True).process(rgb)

    feats = []
    if results.multi_hand_landmarks:
        # Build x_/y_ across ALL landmarks for normalization (global min like your code)
        all_x, all_y = [], []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_x.append(lm.x)
                all_y.append(lm.y)

        if len(all_x) == 0:
            return feats

        min_x, min_y = min(all_x), min(all_y)

        for hand_landmarks in results.multi_hand_landmarks:
            vec = []
            for lm in hand_landmarks.landmark:
                vec.append(lm.x - min_x)
                vec.append(lm.y - min_y)
            feats.append(vec)
    return feats

# ----------------------------
# Dataset building
# ----------------------------
def build_dataset_from_disk(label_map: dict[str, int]):
    data = []
    labels = []

    # Iterate folders
    for lab_name, lab_id in label_map.items():
        folder = DATA_DIR / lab_name
        if not folder.exists():
            continue
        for img_path in folder.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feats_per_hand = extract_features_from_bgr(img)
            # Use the FIRST hand only, to stay consistent and avoid variable length
            if len(feats_per_hand) > 0:
                data.append(feats_per_hand[0])
                labels.append(lab_id)

    return {"data": data, "labels": labels}

def save_label_map(label_map: dict):
    with open(LABELMAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

def load_label_map() -> dict | None:
    if LABELMAP_PATH.exists():
        with open(LABELMAP_PATH, "r") as f:
            return json.load(f)
    return None

# ----------------------------
# Training
# ----------------------------
def train_model(data_dict, n_estimators=100, max_depth=None, random_state=42, test_size=0.2):
    X = np.asarray(data_dict["data"], dtype=object)
    y = np.asarray(data_dict["labels"])

    # Convert ragged to equal-length by padding if needed (shouldn't be needed if we use first hand)
    # But in case of mixed images (no hand vs hand), filter:
    mask = np.array([len(v) > 0 for v in X], dtype=bool)
    X, y = X[mask], y[mask]

    # Convert list of floats -> np arrays with equal length
    # Each should be 42 features for 21 landmarks * 2
    X = np.stack([np.array(v, dtype=np.float32) for v in X], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc

def save_model(model, label_map: dict):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_map": label_map}, f)

def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    # Backward-compat: if label_map missing, load from file or infer
    if "label_map" not in obj:
        lm = load_label_map() or default_label_map_from_folders()
        obj["label_map"] = lm
    return obj

# ----------------------------
# Inference helpers
# ----------------------------
def predict_image(bgr_img: np.ndarray, model, label_map: dict) -> tuple[str | None, np.ndarray]:
    inv_label_map = {v: k for k, v in label_map.items()}
    feats_per_hand = extract_features_from_bgr(bgr_img)
    annotated = bgr_img.copy()

    if len(feats_per_hand) == 0:
        return None, annotated

    # Take first hand (consistent with training)
    vec = np.array(feats_per_hand[0], dtype=np.float32).reshape(1, -1)
    pred_id = int(model.predict(vec)[0])
    pred_label = inv_label_map.get(pred_id, str(pred_id))

    # Draw landmarks for nicer UX
    rgb = bgr_to_rgb(bgr_img)
    results = mp_hands_detector(static=True).process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:1]:
            mp_draw.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(annotated, f"Prediction: {pred_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return pred_label, annotated

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Hand Gesture ML — End-to-End", page_icon="🖐", layout="wide")
st.title("🖐 Hand Gesture Recognition — End-to-End App")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["1) Collect", "2) Build Dataset", "3) Train", "4) Inference"], index=0)
    st.markdown("---")
    st.caption("Tips:\n- Use short, consistent backgrounds\n- Keep hand centered\n- Vary distance & orientation")

# ----------------------------
# 1) Collect
# ----------------------------
if page.startswith("1"):
    st.subheader("1) Collect Dataset")

    ensure_dir(DATA_DIR)
    current_labels = list_labels()
    st.write("Existing labels:", ", ".join(current_labels) if current_labels else "None yet")

    new_label = st.text_input("Label name (e.g., 0, 1, 2 or A, B, L):", value=(current_labels[0] if current_labels else "0"))
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Capture with your camera (browser-friendly):**")
        img_bytes = st.camera_input("Show the gesture and click 'Take Photo'")

        if img_bytes is not None:
            ensure_dir(DATA_DIR / new_label)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = DATA_DIR / new_label / f"{ts}.jpg"
            # Save as captured (RGB) -> convert to BGR for JPEG
            pil_img = Image.open(img_bytes)
            cv2.imwrite(str(save_path), pil_to_bgr(pil_img))
            st.success(f"Saved: {save_path}")

    with col2:
        st.markdown("**Or upload existing images:**")
        files = st.file_uploader("Upload JPG files", type=["jpg", "jpeg"], accept_multiple_files=True)
        if files and st.button("Save uploads to dataset"):
            ensure_dir(DATA_DIR / new_label)
            for f in files:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = DATA_DIR / new_label / f"{ts}.jpg"
                img = load_image_any(f.read())
                cv2.imwrite(str(save_path), img)
            st.success(f"Saved {len(files)} image(s) to data/{new_label}")

    st.info("Repeat for each gesture label you want. Then go to **2) Build Dataset**.")

# ----------------------------
# 2) Build Dataset
# ----------------------------
elif page.startswith("2"):
    st.subheader("2) Build Dataset (feature extraction)")

    # Choose label mapping approach
    st.markdown("**Label mapping** (how folder names map to integer IDs used by the model):")
    default_map = default_label_map_from_folders()
    manual = st.checkbox("Edit label map manually", value=False)

    if manual and default_map:
        edited = {}
        for lab, default_id in default_map.items():
            edited[lab] = st.number_input(f"ID for label '{lab}'", min_value=0, value=int(default_id), step=1)
        # Ensure unique IDs
        if len(set(edited.values())) != len(edited):
            st.error("Label IDs must be unique.")
            st.stop()
        label_map = edited
    else:
        label_map = default_map

    if st.button("Build dataset now"):
        if not label_map:
            st.warning("No labels found in data/. Add images in 'Collect' first.")
        else:
            with st.spinner("Extracting hand landmarks and building dataset..."):
                ds = build_dataset_from_disk(label_map)
                with open(PICKLE_PATH, "wb") as f:
                    pickle.dump(ds, f)
                save_label_map(label_map)
            st.success(f"Dataset saved to {PICKLE_PATH}. Samples: {len(ds['data'])}")

            # Quick summary
            counts = {k: 0 for k in label_map.keys()}
            for y in ds["labels"]:
                # invert to label str
                inv = {v: k for k, v in label_map.items()}
                counts[inv[y]] += 1
            st.write("Samples per label:", counts)

    st.caption("This step mirrors your original `create_dataset.py` preprocessing (translation-invariant landmark coordinates).")

# ----------------------------
# 3) Train
# ----------------------------
elif page.startswith("3"):
    st.subheader("3) Train Model")

    n_estimators = st.slider("RandomForest n_estimators", 50, 500, 200, step=50)
    max_depth = st.selectbox("max_depth", ["None", 8, 12, 16, 24], index=0)
    test_size = st.slider("Test size (validation %)", 10, 40, 20, step=5)
    random_state = st.number_input("random_state", value=42, step=1)

    if st.button("Train now"):
        if not PICKLE_PATH.exists():
            st.warning("No dataset found. Build it in step 2 first.")
        else:
            with open(PICKLE_PATH, "rb") as f:
                data_dict = pickle.load(f)
            lm = load_label_map() or default_label_map_from_folders()
            if not lm:
                st.warning("No label map found.")
                st.stop()
            with st.spinner("Training Random Forest..."):
                clf, acc = train_model(
                    data_dict,
                    n_estimators=n_estimators,
                    max_depth=None if max_depth == "None" else int(max_depth),
                    random_state=int(random_state),
                    test_size=test_size / 100.0
                )
                save_model(clf, lm)
            st.success(f"Training complete. Validation accuracy: {acc:.3f}")
            st.write(f"Model saved to `{MODEL_PATH}`")
            st.json(lm)

# ----------------------------
# 4) Inference
# ----------------------------
elif page.startswith("4"):
    st.subheader("4) Live / Image Inference")

    model_obj = load_model()
    if not model_obj:
        st.warning("No trained model found. Train in step 3.")
        st.stop()

    model = model_obj["model"]
    label_map = model_obj["label_map"]
    st.write("Loaded label map:", label_map)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**A) Single-image inference (camera):**")
        snap = st.camera_input("Take a photo")
        if snap is not None:
            img_bgr = pil_to_bgr(Image.open(snap))
            pred, annotated = predict_image(img_bgr, model, label_map)
            st.image(bgr_to_rgb(annotated), caption=f"Prediction: {pred or 'No hand detected'}")

    with colB:
        st.markdown("**B) Single-image inference (upload):**")
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if up is not None:
            img_bgr = load_image_any(up.read())
            pred, annotated = predict_image(img_bgr, model, label_map)
            st.image(bgr_to_rgb(annotated), caption=f"Prediction: {pred or 'No hand detected'}")

    st.info("For continuous real-time video, run locally and adapt with OpenCV + `st.image()` loop. "
            "Browser-only deployments (Streamlit Cloud) support single-frame via camera/upload reliably.")

