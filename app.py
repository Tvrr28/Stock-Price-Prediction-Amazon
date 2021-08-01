# -*- coding: utf-8 -*-
"""
Created on Sun August 1 10:30:00 2021
@author: TVR Raviteja
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image

pickle_in = open("model.pkl","rb")
linear = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def main():
    st.title("Amazon Inc. Stock Price Prediction")


    
    html_temp = """
    <div style="background-color:#FFD700;padding:10px">
    <h2 style="color:white;text-align:center;">Amazon Close Price Predictor</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)



    Open = st.text_input("Open","Enter a value")
    High = st.text_input("High","Enter a value")
    Low = st.text_input("Low","Enter a value")
    Volume = st.text_input("Volume","Enter a value")
    
    
    compound = st.slider("Compound",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    negative = st.slider("negative",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    neutral = st.slider("neutral",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    positive = st.slider("positive",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    Subjectivity = st.slider("Subjectivity",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    Polarity = st.slider("Polarity",min_value=0.00 , max_value = 1.00 ,step = 0.01)

    af = pd.DataFrame()
    af['compound'] = compound,compound
    af['negative'] = negative,negative
    af['neutral'] = neutral,neutral
    af['positive'] = positive,positive
    af['Open'] = Open,Open 
    af['High'] = High,High
    af['Low'] = Low,Low
    af['Volume'] = Volume,Volume
    af['Subjectivity'] = Subjectivity,Subjectivity
    af['Polarity'] = Polarity,Polarity
    af = np.array(af)
    
    result=""

    if st.button("Predict"):
        result = linear.predict(af)[0]
    st.success('Predicted Close Price : $ {}'.format(result))
    if st.button("About"):
        st.text("Close price of a stock is predicted based on news headlines and historical data using Linear Regression")

if __name__ =='__main__':
    main()
