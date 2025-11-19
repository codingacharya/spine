import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("spine_classifier.h5")

# Put your class names here (same order used during training!)
class_names = ["degenerative", "herniation", "normal"]

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):
    # Convert image to RGB - IMPORTANT FIX
    image = image.convert("RGB")

    # Resize image to model input size
    img = image.resize((224, 224))

    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess for MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)[0]

    # Find class & confidence
    predicted_index = np.argmax(preds)
    predicted_class = class_names[predicted_index]
    confidence = float(preds[predicted_index] * 100)

    return predicted_class, confidence


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Spine Problem Detection")
st.write("Upload an MRI/X-ray image to detect spine conditions.")

uploaded_file = st.file_uploader("Upload Spine Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            predicted_class, confidence = predict_image(img)

        st.success(f"Prediction: **{predicted_class.upper()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
