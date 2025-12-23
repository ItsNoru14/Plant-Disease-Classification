import streamlit as st
import pandas as pd
import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess

st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* Background utama */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f2027,
            #203a43,
            #2c5364
        );
        color: #ffffff;
    }

    /* Container utama */
    .block-container {
        padding: 2.5rem 2.2rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
    }

    /* Judul */
    h1 {
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #e8f5e9;
    }

    /* Subjudul & teks */
    h2, h3, p {
        color: #c8e6c9;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌿 Plant Disease Classification")
st.write(
    """
    Aplikasi ini merupakan model untuk  
    mengklasifikasikan **penyakit pada tanaman** berdasarkan.
    """
)
st.markdown("---")

# LOAD METADATA / CLASS NAMES
SPLIT_DIR = "splits"
ENCODER_DIR = "encoders"
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

# LOAD DATASET
@st.cache_resource
def load_dataset():
    # Load CSV split
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train_encoded.csv"))
    val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "val_encoded.csv"))
    test_df  = pd.read_csv(os.path.join(SPLIT_DIR, "test_encoded.csv"))

    # Load encoder mapping
    label_to_index_path = os.path.join(ENCODER_DIR, "label_to_index.json")
    index_to_label_path = os.path.join(ENCODER_DIR, "index_to_label.json")

    if not os.path.exists(label_to_index_path) or not os.path.exists(index_to_label_path):
        st.error("File encoder tidak ditemukan. Jalankan proses label encoding terlebih dahulu.")
        st.stop()

    with open(label_to_index_path, "r") as f:
        label_to_index = json.load(f)

    with open(index_to_label_path, "r") as f:
        index_to_label = json.load(f)

    return train_df, val_df, test_df, label_to_index, index_to_label

train_df, val_df, test_df, label_to_index, index_to_label = load_dataset()
CLASS_NAMES = [label for idx, label in sorted(index_to_label.items(), key=lambda x: int(x[0]))]
NUM_CLASSES = len(CLASS_NAMES)


# LOAD MODEL
MODEL_DIR = "models"
MODEL_PATHS = {
    "CNN Base": os.path.join(MODEL_DIR, "cnn_base_best.keras"),
    "EfficientNetB0": os.path.join(MODEL_DIR, "efficientnet_best_finetune.keras"),
    "MobileNetV3 Small": os.path.join(MODEL_DIR, "mobilenetv3_best.keras"),
}

@st.cache_resource
def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model {model_name} tidak ditemukan: {model_path}")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

st.subheader("🔍 Pilih Model")
model_choice = st.selectbox(
    "Pilih model yang akan digunakan:",
    list(MODEL_PATHS.keys())
)

model = load_model(model_choice)
st.success(f"{model_choice} berhasil dimuat.")


# INPUT SIZE UNTUK PREPROCESSING
INPUT_SHAPES = {
    "CNN Base": (128, 128),
    "EfficientNetB0": (224, 224),
    "MobileNetV3 Small": (224, 224)
}

selected_input_size = INPUT_SHAPES[model_choice]


# UPLOAD GAMBAR
def upload_image():
    uploaded_file = st.file_uploader(
        "Upload gambar daun (jpg / png)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", width=400)
        return image
    return None

def preprocess_for_model(image: Image.Image, model_name: str):
    input_sizes = {
        "CNN Base": (128, 128),
        "EfficientNetB0": (224, 224),
        "MobileNetV3 Small": (224, 224)
    }
    target_size = input_sizes[model_name]
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    
    if model_name == "CNN Base":
        img_array /= 255.0
    elif model_name == "EfficientNetB0":
        img_array = efficientnet_preprocess(img_array)
    elif model_name == "MobileNetV3 Small":
        img_array = mobilenet_preprocess(img_array)
    else:
        raise ValueError(f"Model {model_name} tidak dikenali")
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_image = upload_image()
if uploaded_image is not None:
    img_array = preprocess_for_model(uploaded_image, model_choice)
    st.success("Gambar berhasil diupload dan diproses.")

# PREDIKSI
def predict_image(model, img_array, index_to_label):
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100
    predicted_class = index_to_label[str(predicted_index)]

    return predicted_class, confidence, predictions[0]

if uploaded_image is not None:
    if st.button("🔮 Prediksi"):
        predicted_class, confidence, all_probs = predict_image(model, img_array, index_to_label)

        st.success(f"🐾 Prediksi penyakit/kelas: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        st.subheader("📈 Probabilitas Semua Kelas")
        prob_dict = {CLASS_NAMES[i]: float(all_probs[i]) for i in range(NUM_CLASSES)}
        st.bar_chart(prob_dict)
        
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #1D2E28;  /* Warna palette yang diminta */
        color: #ffffff;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 9999;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
    }
    </style>
    <div class="footer">
        🌿 Plant Disease Classification &nbsp;|&nbsp; Developed by Dimas Arief Wicaksono &nbsp;|&nbsp; 2025
    </div>
    """,
    unsafe_allow_html=True
)