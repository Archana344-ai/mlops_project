import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Simple ML Model')

feature1 = st.number_input('Feature 1', value=0)
feature2 = st.number_input('Feature 2', value=0)

if st.button('Predict'):
    input_df = pd.DataFrame([[feature1, feature2]], columns=['feature1', 'feature2'])
    prediction = model.predict(input_df)
    st.write('Prediction:', int(prediction[0]))
