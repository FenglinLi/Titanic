# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:51:17 2017

@author: lif8
"""
#set work directory
import os
<<<<<<< HEAD
os.chdir('C:\Users\lif8\Documents\GitHub\Titanic')
#os.chdir('C:\Users\lfl1001\Documents\GitHub\Titanic')
=======
#os.chdir('C:\Users\lif8\Documents\GitHub\Titanic')
os.chdir("C:\\Users\\lfl1001\\Documents\\GitHub\\Titanic")
>>>>>>> e91f9c551d9f719f2c7a02724c84f5ca1191e64f

#import packages
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


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

#create IsAlone feature
for dataset in data_full:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#Fullfill fare in data_test and create Fareband
data_test[ 'Fare' ] = data_test.Fare.fillna( data_test.Fare.mean() )
group_names_fare = ['1', '2', '3', '4']
bins_fare = (-10, 7.91, 14.454, 31, 600)
data_train['FareBand'] = pd.cut(data_train['Fare'], bins_fare, labels=group_names_fare).astype(int)
data_test['FareBand'] = pd.cut(data_test['Fare'], bins_fare, labels=group_names_fare).astype(int)


#Fulfill Embarked and convert to number 
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in data_full:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
for dataset in data_full:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 1, 'Q': 2, 'C': 3} )
    
#extract title based on name feature
for dataset in data_full:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in data_full:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data_full:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#Fulfill cabin feature based on Fare
    
#Drop Cabin, Fare, Ticket, Parch, Name
data_train = data_train.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare', 'Name', 'Parch', 'Age', 'SibSp'], axis=1)
data_test = data_test.drop(['Ticket', 'Cabin', 'Fare', 'Name', 'Parch', 'Age', 'SibSp'], axis=1)

#Fit Model
train_X = data_train.drop('Survived', axis=1)
train_Y = data_train['Survived']

#Split training data 
from sklearn.model_selection import train_test_split

train_XSAll = data_train.drop('Survived', axis=1)
train_YAll = data_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

test_X  = data_test.drop("PassengerId", axis=1).copy()

<<<<<<< HEAD
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}
=======
clf = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(train_X, train_Y)

clf.fit(train_X, train_Y)

predictions = clf.predict(test_X)
print(accuracy_score(test_Y, predictions))
>>>>>>> e91f9c551d9f719f2c7a02724c84f5ca1191e64f

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_Y)
acc_random_forest = round(random_forest.score(train_X, train_Y) * 100, 2)
pred_Y = random_forest.predict(test_X)

submission_my = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],
        "Survived": pred_Y
    })

submission_my = submission_my.to_csv('submission_my.csv', index=False)



