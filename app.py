# -*- coding: utf-8 -*-
"""
Created on Sun August 1 10:30:00 2021
@author: TVR Raviteja
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st
import sklearn

from PIL import Image

pickle_in = open("model.pkl","rb")
linear = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def main():
    st.title("Amazon Inc. Stock Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Amazon Close Price Predictor</h2>
    </div>
    """
    page_bg_img = """
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1500&q=80");
    background-size: cover;
    }
    </style>
    """
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(html_temp,unsafe_allow_html=True)
    Open = st.text_input("Open","Enter a value")
    High = st.text_input("High","Enter a value")
    Low = st.text_input("Low","Enter a value")
    Volume = st.text_input("Volume","Enter a value")
    compound = st.text_input("compound","Enter a value based on recent headlines")
    negative = st.text_input("negative","Enter a value based on recent headlines")
    neutral = st.text_input("neutral","Enter a value")
    positive = st.text_input("positive","Enter a value")
    Subjectivity = st.text_input("Subjectivity","Enter a value")
    Polarity = st.text_input("Polarity","Enter a value")

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
    st.success('Predicted Close Price : {}'.format(result))
    if st.button("About"):
        st.text("Close price of a stock is predicted based on news headlines and historical data using Linear Regression")

if __name__ =='__main__':
    main()
