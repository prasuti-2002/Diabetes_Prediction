# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:26:36 2023

@author: dell
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('E:/diabetes/trained_model.sav','rb'))

#creating a fumction for Prediction
def diabetes_prediction(input_data):
    
     #Changing the input data to numpy Array
     input_data_as_numpy_array=np.asarray(input_data)
     #reshape the array as we are predicting for one instance
     input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

     #Standardized the input data

     prediction=loaded_model.predict(input_data_reshaped)
     print(prediction)


     if(prediction[0]==0):
         return'The person is not diabetic'
     else:
         return'The person is diabetic'
         
         
def main():
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    Pregnancies= st.text_input('Number of Pregnancies')
    Glucose= st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('Skin Thickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('Diabetes pedigree Function Value')
    Age= st.text_input('Age of the person')
    
    #Code for the Prediction
    diagnosis=''
    #Creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)    
        
if __name__ == "__main__":
    main()
        
        
        