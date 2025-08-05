import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Model ko load karo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

model = load_model()

# Class labels
class_labels = ['Cat', 'Dog']

st.title("Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or a dog to see the prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Image ko display karo
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Image ko model ke liye tayyar karo
    img = image_to_predict.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # Normalize karo

    # Prediction karo
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    # Result show karo
    if confidence > 0.5:
        predicted_class = 'Dog'
        confidence_percent = confidence * 100
    else:
        predicted_class = 'Cat'
        confidence_percent = (1 - confidence) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence_percent:.2f}%")