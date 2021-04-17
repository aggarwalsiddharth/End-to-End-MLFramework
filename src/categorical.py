import pandas as pd
from sklearn import preprocessing

'''
- label encoding
- one hot encoding
- binarization

'''

class CategoricalFeatures:
    def __init__(self,df,categorical_features, encoding_type, handle_na = False):

        '''
        df : dataframe

        categorical_features: Name of cols that are categorical

        encoding_type: label_encoding, binarization, one_hot

        '''
        self.df = df

        self.output_df = self.df.copy(deep=True)

        self.categorical_features = categorical_features
        self.encoding_type = encoding_type
        self.label_encoders = dict()
        self.handle_na = handle_na

        if self.handle_na == True:
            for c in self.categorical_features:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna('-999999')


    def _label_encoding(self):

        for c in self.categorical_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)

            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)

            self.label_encoders[c] = lbl

        return self.output_df


    def transform(self):
        if self.encoding_type == 'label_encoding':
            return self._label_encoding()

        else:
            raise Exception(" Encoding type not found")


if __name__=='__main__':

    df = pd.read_csv("../input/train_categorical.csv")
    cols = [c for c in df.columns if c not in ["id","target"]]
    # print(cols)

    cat_feats = CategoricalFeatures(df, 
                                    categorical_features = cols,
                                    encoding_type= "label_encoding",
                                    handle_na= True
                                    )

    output_df = cat_feats.transform()

    print(output_df.head())
    # print(output_df[output_df.isnull().any(axis=1)].head())