import os
import pandas as pd 
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")


FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
0: [1, 2, 3, 4],
1: [0, 2, 3, 4],
2: [0, 1, 3, 4],
3: [0, 1, 2, 4],
4: [0, 1, 2, 3]

}


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.Survived.values
    yvalid = valid_df.Survived.values

    train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin','kfold','Survived'],axis =1)
    valid_df = valid_df.drop(['PassengerId','Name','Ticket','Cabin','kfold','Survived'],axis =1)

    valid_df = valid_df[train_df.columns]


    train_df['Family'] = train_df['Parch'] + train_df['SibSp'] + 1
    train_df.drop(['Parch','SibSp'],axis = 1,inplace = True)

    valid_df['Family'] = valid_df['Parch'] + valid_df['SibSp'] + 1
    valid_df.drop(['Parch','SibSp'],axis = 1,inplace = True)

    df_test['Family'] = df_test['Parch'] + df_test['SibSp'] + 1
    df_test.drop(['Parch','SibSp'],axis = 1,inplace = True)

    # Imputing missing values

    for c in train_df.columns[train_df.isnull().any()]:
        if train_df[c].dtype!=object:
            print(c +'-not string')
            train_df[c].fillna(train_df[c].median(), inplace = True)
            valid_df[c].fillna(valid_df[c].median(), inplace = True)

        else:
            print(c + '-string')
            train_df[c].fillna(train_df[c].mode()[0], inplace = True)
            valid_df[c].fillna(valid_df[c].mode()[0], inplace = True)
    
    
    
    
    # Label encoding


    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist()+ df_test[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())

        label_encoders[c]=lbl



    ## Now data is ready to train

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df,ytrain)

    preds = clf.predict_proba(valid_df)[:,1]

    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders,f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf,f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns,f"models/{MODEL}_{FOLD}_columns.pkl")


    

