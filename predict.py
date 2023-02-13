from model import train

import xgboost as xgb
import streamlit as st
import pandas as pd
import pickle



@st.cache_data

def predict(data):
    #file = open()
    model = pickle.load(open(r"C:\Users\indra\Documents\Model Deployment\model_pkl",'rb'))
    

    trf = train()

    data_trf = trf.transform(data)

    pred = model.predict(data_trf)
    
    return pred