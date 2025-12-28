import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('artifacts/model.h5')

with open('artifacts/label_encoder_gender.pkl','rb') as file:
    label_enocder_gender=pickle.load(file)
with open('artifacts/ohe_geo.pkl','rb') as file:
    ohe_geo=pickle.load(file)
with open('artifacts/scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
## Streamlit App
st.title("Customer Churn Prediction")

### User Input
with st.form('Churn_form'):
    geography = st.selectbox('Geography',ohe_geo.categories_[0])
    gender = st.selectbox('Gender',label_enocder_gender.classes_)
    age = st.slider('Age',18,100)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated salary')
    tenure = st.slider('Tenure',0,10)
    number_of_products = st.slider('Products',1,4)
    has_cr_card = st.selectbox('Has Credit Card',[0,1])
    is_active_member = st.selectbox('Is active member',[0,1])
    
    submit = st.form_submit_button('Predict')
    
if submit:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_enocder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [number_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = ohe_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

    input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

    ## Scale
    input_scaled = scaler.transform(input_df)

    ## Predict Churn 

    prediction = model.predict(input_scaled)[0][0]
    print(prediction)
    if prediction > 0.5:
        st.write("Customer is likely to churn")
    else:
        st.write("Customer is safe")