#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:13:59 2021

@author: xihongshijidanzhajiangmian
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


x=dataset.drop(columns='HeartDisease')
y=dataset['HeartDisease']
#find the target column 'heart disease' and remove it from the original list

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)
model=DecisionTreeClassifier()
model.fit(x_train, y_train)#create model
y_pred=model.predict(x_test)

score=accuracy_score(y_pred, y_test)#accuracy of the predict
print(score)

y_pred_prob=model.predict_proba(x_test)
b=pd.DataFrame(y_pred_prob, columns=['heart disease','no heart disease'])


auc_score=roc_auc_score(y_test, y_pred_prob[:,1])


model.feature_importances_#the improtance of each variable in the dataset
#转化为文本？











