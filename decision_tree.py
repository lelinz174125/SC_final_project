#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:13:59 2021

@author: xihongshijidanzhajiangmian
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


dataset=pd.DataFrame(pd.read_csv('heart_data_addressed.csv'))
a=dataset.head(5)
print(a)#check if it is the desired dataset format
#print(dataset)


x=dataset.drop(columns='HeartDisease')#dataset except target
y=dataset['HeartDisease']#target
#find the target column 'heart disease' and remove it from the original list

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)#data split
model=DecisionTreeClassifier(random_state=0)
model.fit(x_train, y_train)#training
y_pred=model.predict(x_test)

score=accuracy_score(y_pred, y_test)
print(score)#accurancy of the model

y_pred_proba=model.predict_proba(x_test)
auc_score=roc_auc_score(y_test, y_pred_proba[:,1])
print(auc_score)

fpr, tpr, thres=roc_curve(y_test, y_pred_proba[:,1])
plt.plot(fpr,tpr)
plt.show()#ROC curve


b=model.feature_importances_#the improtance of each variable in the dataset
print(b)
#ST_Slpo_Llat is the most improtant feature with a improtance of 0.4227









