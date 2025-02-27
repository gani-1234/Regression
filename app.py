import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# load the model
model=load_model('model.keras')
# load the scaler and encoders
with open('label_encoder.pkl','rb')as file:
    label_encoder=pickle.load(file)
with open('OneHotEncoder_geo.pkl','rb')as file:
    OneHotEncoder_geo=pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

# i/p data 
st.title("Estimated Salary Prediction")
CustomerId=st.number_input('CustomerId :')
CreditScore=st.slider('CreditScore :',100,850)
Gender=st.selectbox('Gender :' ,label_encoder.classes_)
Age=st.slider('Age :',18,100)
Tenure=st.slider('Tenure :',1,5)
Balance=st.number_input('Balance :')
Geography=st.selectbox('Geography :',OneHotEncoder_geo.categories_[0])
NumOfProducts=st.number_input('NoOfProducts :')
HasCrCard=st.selectbox('HasCrCard (1 for yes) :',[1,0])
IsActiveMember=st.selectbox('IsActiveMember (1 for yes) :',[1,0])
EstimatedSalary=st.number_input('EstimatedSalary :')
Exited=st.selectbox('Exited (1 for yes) :',[1,0])

input_data=pd.DataFrame({
    'CustomerId':[CustomerId],
    'CreditScore':[CreditScore],
    'Gender':[Gender],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Balance],
    'Geography':[Geography],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
    # 'Exited':[Exited]
})
input_data=pd.DataFrame(input_data)
# labelencder for gender column
input_data['Gender']=label_encoder.transform(input_data['Gender'])

# onehot_encoder for Gegrahy clumn
geo_encoder=OneHotEncoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoder=pd.DataFrame(geo_encoder,columns=OneHotEncoder_geo.get_feature_names_out(['Geography']))

# combine onehot_encoder data with original data
input_data=pd.concat([input_data.drop('Geography',axis=1),geo_encoder],axis=1)


input_data_scaled=scaler.transform(input_data)
input_data.columns=input_data.columns.astype(str)
# compile the model
prediction=model.predict(input_data_scaled)
prediction_probability=prediction[0][0]

st.write('Salary probability is :',prediction_probability)


