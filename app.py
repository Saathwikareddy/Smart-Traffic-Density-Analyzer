import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("traffic_sign_model.keras")

# Class names
classes = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    14: "Stop Sign",
    17: "No Entry"
}

st.set_page_config(page_title="Traffic Sign Detection")

st.title("🚦 Traffic Sign Detection System")
st.write("Upload a traffic sign image to detect the sign.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((32, 32))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction)

    st.subheader("Prediction")

    if class_id in classes:
        st.success(
            f"Detected Sign: {classes[class_id]}"
        )
    else:
        st.warning(
            f"Detected Class ID: {class_id}"
        )

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    # Analytics Dashboard
    st.subheader("📊 Traffic Analytics")

    st.metric(
        "Predicted Class ID",
        class_id
    )

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )
