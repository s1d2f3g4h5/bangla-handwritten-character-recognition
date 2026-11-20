
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageOps
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bangla Character Recognition",
    page_icon="বাংলা",
    layout="centered"
)

APP_DIR = Path(__file__).parent

MODEL_PATH = APP_DIR / "bangla_hcr_final.keras"
MAPPING_PATH = APP_DIR / "class_mapping.csv"

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

@st.cache_data
def load_mapping():
    return pd.read_csv(
        MAPPING_PATH,
        keep_default_na=False
    )

# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess_image(image):

    gray = np.array(
        ImageOps.exif_transpose(image).convert("L")
    )

    # Blur
    blurred = cv2.GaussianBlur(
        gray, (3, 3), 0
    )

    # Detect foreground
    border = np.concatenate([
        blurred[0, :],
        blurred[-1, :],
        blurred[:, 0],
        blurred[:, -1]
    ])

    if np.median(border) > 127:
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Remove very small components
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary, 8
        )
    )

    clean = np.zeros_like(binary)

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            clean[labels == i] = 255

    # Find character bounding box
    ys, xs = np.where(clean > 0)

    if len(xs) == 0:
        raise ValueError("No handwriting detected.")

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    pad_x = max(5, int((x2-x1) * 0.12))
    pad_y = max(5, int((y2-y1) * 0.12))

    x1 = max(0, x1-pad_x)
    x2 = min(clean.shape[1], x2+pad_x)

    y1 = max(0, y1-pad_y)
    y2 = min(clean.shape[0], y2+pad_y)

    cropped = clean[y1:y2, x1:x2]

    # Resize with aspect ratio
    h, w = cropped.shape
    scale = 52 / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Center on 64x64 canvas
    canvas = np.zeros((64, 64), dtype=np.uint8)

    sx = (64-new_w)//2
    sy = (64-new_h)//2

    canvas[
        sy:sy+new_h,
        sx:sx+new_w
    ] = resized

    tensor = canvas.astype(
        np.float32
    ) / 255.0

    return tensor[..., None], canvas

# ==========================================
# MAIN USER INTERFACE
# ==========================================

st.title("বাংলা অক্ষর চিনুন")
st.subheader("Bangla Handwritten Character Recognition")

st.write(
    "আপনার হাতে লেখা একটি বাংলা অক্ষরের ছবি Upload করুন। "
    "আমাদের AI Model সেটি শনাক্ত করার চেষ্টা করবে।"
)

uploaded_file = st.file_uploader(
    "Upload your handwriting",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Handwriting",
        use_container_width=True
    )

    if st.button(
        "অক্ষর শনাক্ত করুন",
        type="primary",
        use_container_width=True
    ):

        try:

            model = load_model()
            mapping = load_mapping()

            input_tensor, processed = preprocess_image(image)

            batch = np.expand_dims(
                input_tensor, axis=0
            )

            prediction = model.predict(
                batch,
                verbose=0
            )[0]

            top_indices = np.argsort(
                prediction
            )[-5:][::-1]

            # Mapping lookup
            labels = dict(
                zip(
                    mapping["class_id"].astype(int),
                    mapping["unicode_character"].astype(str)
                )
            )

            st.divider()

            st.subheader("Prediction Results")

            best_index = int(top_indices[0])
            best_class = best_index + 1
            best_label = labels.get(
                best_class, ""
            ).strip()

            st.image(
                processed,
                caption="Processed CNN Input",
                width=180
            )

            st.metric(
                "Predicted Character",
                best_label if best_label else f"Class {best_class}",
                f"{prediction[best_index]*100:.2f}% confidence"
            )

            st.subheader("Top 5 Predictions")

            for idx in top_indices:

                class_id = int(idx) + 1
                label = labels.get(
                    class_id, ""
                ).strip()

                confidence = float(
                    prediction[idx] * 100
                )

                name = (
                    label if label
                    else f"Class {class_id}"
                )

                st.write(
                    f"**{name}** — {confidence:.2f}%"
                )

                st.progress(confidence / 100)

            if prediction[best_index] < 0.50:
                st.warning(
                    "Model is uncertain. "
                    "Try uploading a clearer image."
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {str(e)}"
            )

st.divider()

st.caption(
    "Bangla HCR | Deep Learning Project | "
    "TensorFlow + CNN + Streamlit"
)
