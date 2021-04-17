import os
import pandas as pd 
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")

MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["PassengerId"].values
    predictions = None

    for FOLD in range(5):
        #print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))


        for c in df.columns[df.isnull().any()]:
            if df[c].dtype!=object:
                df[c].fillna(df[c].median(), inplace = True)

            else:
                df[c].fillna(df[c].mode()[0], inplace = True)
        
        
        df['Family'] = df['Parch'] + df['SibSp'] + 1
        df.drop(['Parch','SibSp'],axis = 1,inplace = True)
        print('hi1')
        
        for c in cols:
            print('hi2')
            lbl = encoders[c]
            df.loc[:,c]= lbl.transform(df[c].values.tolist())


        clf = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        df = df[cols]



        preds = clf.predict_proba(df)[:,1]
        if FOLD ==0:
            predictions = preds
        
        else:
            predictions += preds

    predictions/=5

    ## to make it 0 and 1
    predictions = np.array(predictions)
    predictions = (predictions>=0.5).astype(int)

    sub = pd.DataFrame(np.column_stack((test_idx,predictions)),columns=["PassengerId","Survived"])
    return sub
    


if __name__ == "__main__":

    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv",index=False)


    

