import streamlit as st
from predict import predict
import numpy as np
import pandas as pd
import time

st.title("Diamond Price Predictor: A Regression Problem")
st.write("This is a ML powered price predictor. For this project, Extreme Gradient Boosting has been used")
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.write("*Please enter the characteristics of the diamond:* ")

carat = np.array(st.number_input("Carat: ",min_value = 0.1, max_value=10.0,value=1.0))
table = np.array(st.number_input("Table: ", min_value=31.0,max_value=70.0,value=51.0))

cut = np.array(st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']))

color = np.array(st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D']))

clarity = np.array(st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']))

depth = np.array(st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0))

table = np.array(st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0))

x = np.array(st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0))

y = np.array(st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0))

z = np.array(st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0))



data = np.column_stack((carat,depth,table,x,y,z,cut, color,clarity))
df = pd.DataFrame(data,columns=["carat", "depth", "table", "x", 'y', 'z', 'cut', 'color', 'clarity'])
if st.button("Predict Price"):
    time.sleep(2)
    with st.progress(value=0, text="Starting the prediction engine..."):
        #time.sleep(1)
        st.progress(value=25,text="Processing the user request...")
        time.sleep(2)
        st.progress(value=50,text="Prediction engine running...")
        time.sleep(3)
        price = predict(df)
        st.progress(value=90,text="Processing the output for user...")
        st.progress(value=100,text="Completed!")
        time.sleep(2)

        
        

    pred_price = list(price)
    st.write("The price of the diamond is: ",pred_price[0].round(2),"$")
    st.success("Price predicted succefully!")
    #st.snow()
