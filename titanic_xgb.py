# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:07:23 2017

@author: lfl1001
"""
import os
#os.chdir('C:\Users\lif8\Documents\GitHub\Titanic')
os.chdir('C:\Users\lfl1001\Documents\GitHub\Titanic')

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])