import streamlit as st 
import numpy as np 
import pickle 

# using saved model 
with open('scaler.pkl', 'rb') as scaler_file: 
    loaded_scaler = pickle.load(scaler_file)
with open('best_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

st.title('SUV CAR PURCHASING PREDICTION')
st.write('This app predicts whether a customer will purchase an SUV car based on their age, Gender and Salary')

# User input 
gender = st.selectbox(label='Select your gender', options=['Male', 'Female'])
if gender == 'Male':
    gender_value = 0
else:
    gender_value = 1
age = st.slider(label='Select age', min_value=18, max_value=100,value=20,step=1)
salary = st.number_input(label="Enter your salary")


# Prediction : 
if st.button(label='PREDICT'):
    input_data = [[gender_value, age, salary]]
    scaled_input = loaded_scaler.transform(input_data)
    model_prediciton = loaded_model.predict(scaled_input)
    # Display answer 
    if model_prediciton == 1:
        st.write("This person will buy the SUV car")
    else:
        st.write("This person will not buy the SUV car")

