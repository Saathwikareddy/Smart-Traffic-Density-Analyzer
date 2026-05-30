import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# Load model
model = joblib.load("traffic_model.pkl")

st.set_page_config(page_title="Traffic Sign Detection", layout="centered")

st.title("🚦 Traffic Sign Detection")
st.write("Upload a traffic sign image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = np.array(image)

    img = cv2.resize(img, (32, 32))

    img = img / 255.0

    img = img.flatten()

    img = img.reshape(1, -1)

    prediction = model.predict(img)[0]

    traffic_signs = {
        0: "Speed Limit",
        14: "Stop Sign",
        17: "No Entry"
    }

    result = traffic_signs.get(
        prediction,
        f"Traffic Sign Class {prediction}"
    )

    st.success(f"Prediction: {result}")
