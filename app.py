import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

st.title("Predicting Life Satisfaction based on GDP per Capita.")
st.write("This Machine Learning model will make a prediction for the Life Satisfaction (1-10) of a country based on it's GDP per Capita.")
st.write("Two values for you to test: ")
st.write("Belgium's GDP per Capita: 53659.317, original value: 6.8")
st.write("Colombia's GDP per Capita: 6971.669, original value: 5.7")

# Load the model
model = joblib.load('model.pkl')

# Ask input value 
st.text_input("GDP per Capita value: ", key="value")

if st.session_state.value:
    # converting input to float
    prediction_value = float(st.session_state.value)
    # storing in 2D array
    data = np.array([[prediction_value]])
    # Make predictions
    predictions = model.predict(data)
    # Print the predictions
    print(predictions) # terminal
    st.write("Life Satisfaction Prediction: ", predictions)

if st.checkbox('Show data'):
    training_data = pd.read_csv("training_data.csv", index_col=0)
    training_data

if st.checkbox('Show me valuable Life Satisfaction advice'):
    st.write("Best Pain au Chocolate I every had.")
    map_data = pd.DataFrame({
        'lat': [50.880660],
        'lon': [4.695650]
    })
    st.map(map_data)