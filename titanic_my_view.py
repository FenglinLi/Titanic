# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 14:22:53 2017

@author: lif8
"""
import os
os.chdir('C:\Users\lif8\Documents\GitHub\Titanic')

#import packages
import pandas as pd
import seaborn as sns
import numpy as np


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_full = [data_train, data_test]


#data at a glance
data_train.head()
data_train.shape
data_test.shape

list(data_train)
data_train.info()

#Pclass with Survived

data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x="Pclass", y="Survived", hue="Pclass", data=data_train);

#Sex with Survived - nomalize 
data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x="Sex", y="Survived", hue="Sex", data=data_train);

#Age with Survied

data_train['AgeBand'] = pd.cut(data_train['Age'], 5)

bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
data_train['AgeBand'] = pd.cut(data_train['Age'], bins, 8)

sns.barplot(x='AgeBand', y="Survived", hue='AgeBand', data=data_train);

#SibSp with Survived
data_train.SibSp.describe()
sns.barplot(x='SibSp', y="Survived", hue='SibSp', data=data_train);

#Parch with Survived
data_train.Parch.describe()
sns.barplot(x='Parch', y="Survived", hue='Parch', data=data_train);

#Fare with Survived
data_train.Fare
fare_c = pd.cut(data_train.Fare,5)
fare_c

sns.barplot(x='Fare', y="Survived", data=data_train);

#cabin with survived
sns.barplot(x='Cabin', y="Survived", data=data_train);
data_train.Cabin.describe()

#Embarked with Survived
sns.barplot(x='Embarked', y="Survived", data=data_train);

