# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:51:17 2017

@author: lif8
"""
#set work directory
import os
os.chdir('C:\Users\lif8\Documents\GitHub\Titanic')

#import packages
import pandas as pd
import seaborn as sns
import numpy as np

#Loading Data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_full = [data_train, data_test]


#Sex to number
for dataset in data_full:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#Fullfill Ages and create AgeBand, '1=Baby', '2=Child', '3=Teenager', '4=Student', '5=Young Adult', '6=Adult', '7=Senior'
data_train[ 'Age' ] = data_train.Age.fillna( data_train.Age.mean() )
data_test[ 'Age' ] = data_test.Age.fillna( data_test.Age.mean() )
bins = (0, 5, 12, 18, 25, 35, 60, 120)
group_names = ['1', '2', '3', '4', '5', '6', '7']
data_train['AgeBand'] = pd.cut(data_train['Age'], bins, labels=group_names).astype(int)
data_test['AgeBand'] = pd.cut(data_test['Age'], bins, labels=group_names).astype(int)


#Combine SibSp and Parch into FamilySize feature
for dataset in data_full:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


#Fullfill fare in data_test and create Fareband
data_test[ 'Fare' ] = data_test.Fare.fillna( data_test.Fare.mean() )