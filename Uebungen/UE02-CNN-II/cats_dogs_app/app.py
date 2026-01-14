import streamlit as st
import tensorflow as tf
from tensorflow import keras  # Sicherer Import für Keras
from PIL import Image
import numpy as np
import os


@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "model.keras")
    return keras.models.load_model(model_path)
model = load_model()


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.title("Katzen & Hunde Klassifizierung")
st.write("Lade ein Bild hoch, um es zu klassifizieren.")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    if st.button("Klassifizieren"):
        with st.spinner("Modell rechnet..."):
            x = preprocess_image(image)
            prediction = model.predict(x)[0][0]
            if prediction < 0.5:
                label = "Katze"
            else:
                label = "Hund"

            st.subheader(f"Ergebnis: {label}")
            st.write(f"Wahrscheinlichkeit für {label}: {float(prediction):.4f}")