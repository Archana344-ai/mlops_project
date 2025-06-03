import streamlit as st
import pickle
import numpy as np

# Load model and label names
with open('artifacts/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/target_names.pkl', 'rb') as f:
    target_names = pickle.load(f)

st.title("🌸 Iris Flower Classifier")

st.write("Enter the flower's features below:")

sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)

if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"🌼 Predicted Species: {target_names[prediction]}")
