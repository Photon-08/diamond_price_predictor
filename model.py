import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder,LabelBinarizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import pickle 
def train():

    df = pd.read_csv(https://github.com/Photon-08/diamond_price_predictor/blob/main/diamonds.csv)
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
    return pipe
train()

