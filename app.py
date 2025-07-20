import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('Garbage Classification App')
st.subheader('Upload an image to classify it into categories')
st.text('This app classifies images into categories: cardboard, glass, metal, paper, plastic, and trash.')
model = load_model('Image_clasification.keras')

data_cat = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
img_height = 180
img_width = 180
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = tf.keras.utils.load_img(uploaded_file, target_size=(img_width, img_height))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    st.image(image)
    st.write(f"This image most likely belongs to {data_cat[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence.")