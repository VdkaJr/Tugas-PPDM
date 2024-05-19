import streamlit as st
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib
import pandas as pd

# Fungsi untuk ekstraksi fitur
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Pastikan glcm adalah array 4-dimensi
    if glcm.ndim != 4:
        raise ValueError("GLCM array must be 4-dimensional")
    
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    ASM = graycoprops(glcm, 'ASM').mean()
    energy = graycoprops(glcm, 'energy').mean()
    return [dissimilarity, correlation, ASM, energy]

model = joblib.load('best_model.pkl')

def get_fruit_icon(fruit):
    icons = {
        'apple': '🍎',
        'banana': '🍌',
        'orange': '🍊'
    }
    return icons.get(fruit, '')

st.set_page_config(page_title="Fruit Classification", page_icon="🍏")
st.title("Welcome to Fruit Classification Web")
st.caption("This web-based application classifies images of fruits into categories such as apple, banana, and orange. Please upload images with clear visibility.")

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    button = st.button("Classify")
    if button:
        results = []
        for image_file in uploaded_file:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            features = extract_features(opencv_image)
            prediction = model.predict([features])
            results.append(prediction[0])

        for i, (image_file, fruit) in enumerate(zip(uploaded_file, results)):
            st.image(image_file, caption=f'Predicted: {fruit} {get_fruit_icon(fruit)}', use_column_width=True)
            st.markdown(f"<h3 style='text-align: center;'>Predicted: {fruit} {get_fruit_icon(fruit)}</h3>", unsafe_allow_html=True)