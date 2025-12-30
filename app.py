import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load model
model = tf.keras.models.load_model('model.h5')

with open('artifacts/label_encoder_gender.pkl','rb') as File:
    label_encode = pickle.load(File)
    
with open('artifacts/onehot_encoder_geo.pkl','rb') as File:
    onehot_encode = pickle.load(File) 
    
with open('artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
    
# Streamlit section
st.set_page_config(page_title="Customer Input Form", layout="centered")

st.title("🏦 Bank Customer Details")

with st.form("customer_form"):
    st.subheader("Enter Customer Information")

    CreditScore = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=600
    )

    Geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    Age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=40
    )

    Tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=10,
        value=3
    )

    Balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=60000.0
    )

    NumOfProducts = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2
    )

    HasCrCard = st.selectbox(
        "Has Credit Card?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    IsActiveMember = st.selectbox(
        "Is Active Member?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    EstimatedSalary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

    submitted = st.form_submit_button("Submit")

if submitted:
    input_data = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }

    st.success("Form submitted successfully!")

    ## Geography
    geo_encoded = onehot_encode.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode.get_feature_names_out(['Geography']))

    input_df=pd.DataFrame([input_data])
    input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

    ## Gender
    input_df['Gender']=label_encode.transform(input_df['Gender'])

    ## Scale the input
    input_scaled=scaler.transform(input_df)

    ## Prediction logic
    predict = model.predict(input_scaled)
    prediction = predict[0][0]
    if prediction > 0.5:
        text = 'The customer is likely to churn.'
    else:
        text = 'The customer is not likely to churn.'
        


    st.write(text)
    
    