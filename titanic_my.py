# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:51:17 2017

@author: lif8
"""
#set work directory
import os
#os.chdir('C:\Users\lif8\Documents\GitHub\Titanic')
os.chdir('C:\Users\lfl1001\Documents\GitHub\Titanic')

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
bins_age = (0, 5, 12, 18, 25, 35, 60, 120)
group_names_age = ['1', '2', '3', '4', '5', '6', '7']
data_train['AgeBand'] = pd.cut(data_train['Age'], bins_age, labels=group_names_age).astype(int)
data_test['AgeBand'] = pd.cut(data_test['Age'], bins_age, labels=group_names_age).astype(int)


#Combine SibSp and Parch into FamilySize feature
for dataset in data_full:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


#Fullfill fare in data_test and create Fareband
data_test[ 'Fare' ] = data_test.Fare.fillna( data_test.Fare.mean() )
group_names_fare = ['1', '2', '3', '4']
bins_fare = (0, 10, 30, 100, 300)
data_train['FareBand'] = pd.cut(data_train['Fare'], bins_fare, labels=group_names_fare).astype(int)
data_test['FareBand'] = pd.cut(data_test['Fare'], bins_fare, labels=group_names_fare).astype(int)

#Fullfill Embarked and convert to number 
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in data_full:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
for dataset in data_full:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 1, 'Q': 2, 'C': 3} ).astype(int)