import streamlit as st
import pickle
import numpy as np
import pandas as pd
pipe = pickle.load(open("pipe.pkl", "rb"))
df=pickle.load(open("new_df.pkl","rb"))
st.title("Laptop Price Predictor")
#Now we will take user input one by one as per our dataframe
proc= st.selectbox("Processor", df['Processor'].unique())
oss= st.selectbox("Operating System", df['Operating_System'].unique())
rams= st.selectbox("RAM Size", df['RAM_Size'].unique())
ramt= st.selectbox("RAM Type", df['RAM_Type'].unique())
store= st.selectbox("Storage", df['Storage'].unique())
#Prediction
if st.button('Predict the Price'):
    query = np.array([proc,oss,rams,ramt,store])
    query = query.reshape(1,-1)
    p = pipe.predict(query)[0]
    result = np.exp(p)
    st.subheader("Predicted Prize : ")
    st.subheader(":red[₹{}]".format(result.round(2)))