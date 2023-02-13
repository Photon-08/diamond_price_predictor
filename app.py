import streamlit as st
#from predict import predict
import numpy as np
import pandas as pd
import time

#training
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder,LabelBinarizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import pickle 
def train():

    df = pd.read_csv("https://drive.google.com/file/d/16XcJAgr-ChvIm9A4h2oxppglkPu74-8v/view?usp=share_link")
    #print(df["table"].unique())

    df = df.drop(df[df["x"]==0].index)
    df = df.drop(df[df["y"]==0].index)
    df = df.drop(df[df["z"]==0].index)

    y = df["price"].copy()
    X = df.drop(["price"],axis=1)

    pipe = ColumnTransformer([("num_trf",StandardScaler(),["carat","depth","table","x","y","z"]),
    ("cat_trf",OrdinalEncoder(),["cut","color","clarity"])])
    #print(X)


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=42)

    pipe.fit(X_train)
    X_train_trf = pipe.transform(X_train)
    X_test_trf = pipe.transform(X_test)

    model = XGBRegressor(random_state=42)
    model.fit(X_train_trf,y_train)

    print(r2_score(y_train,model.predict(X_train_trf)))
    print(r2_score(y_test,model.predict(X_test_trf)))

    with open('model_pkl', 'wb') as files:
        pickle.dump(model, files)
    #print(pipe.n_features_in_)
    #print(pipe.get_feature_names_out())
    return pipe, model
train()

#testing 
#from model import train

import xgboost as xgb
import streamlit as st
import pandas as pd
import pickle

@st.cache_data

def predict(data):
    #file = open()
    model, trf = train()
    model = pickle.load(open(model,'rb'))
    

    

    data_trf = trf.transform(data)

    pred = model.predict(data_trf)
    
    return pred







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
