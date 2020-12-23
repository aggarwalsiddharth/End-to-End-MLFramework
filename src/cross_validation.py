'''
Types of problems:

-binary classification
-multiclass classification
-multi label classification
-single column regression
-multi column regression
-holdout

'''

import pandas as pd
from sklearn import model_selection



class CrossValidation:

    def __init__(self,
                df,
                target_cols,
                problem_type='binary_classification',
                num_fold = 5,
                shuffle = True,
                random_state = 42,
                ):

        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_fold = num_fold
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe['kfold'] = -1

    def split(self):

        if self.problem_type in ['binary_classification','multiclass_classification']:
            target = self.target_cols[0]
            kf = model_selection.StratifiedKFold(n_splits = self.num_fold,
                                                 shuffle = False,
                                                 random_state = self.random_state)

            for fold, (train_idx,val_idx) in enumerate(kf.split(X=self.dataframe,y = self.dataframe[target].values)):
                self.dataframe.loc[val_idx,'kfold'] = fold



        elif self.problem_type in ['single_col_regression','muti_col_regression']:
            kf = model_selection.KFold(n_splits=self.num_fold)
            for fold, (train_idx,val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx,'kfold'] = fold

        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage/100)

            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples,"kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:,"kfold"] = 1


        else:
            raise Exception('Problem Type not found')

        
        return self.dataframe


if __name__ == "__main__":

    df = pd.read_csv("../input/train.csv")

    cv = CrossValidation(df,target_cols = ['Survived'])

    df_split = cv.split()

    print(df_split.head())






