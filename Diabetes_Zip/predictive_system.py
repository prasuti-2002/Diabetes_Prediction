# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
#loading the saved model
loaded_model=pickle.load(open('E:/diabetes/trained_model.sav','rb'))

input_data=(1,85,66,29,0,26.6,0.351,31)
#Changing the input data to numpy Array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#Standardized the input data

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)


if(prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')