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
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


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
#get 0.81 around

fpr, tpr, thres=roc_curve(y_test, y_pred_proba[:,1])
plt.plot(fpr, tpr, lw=1)
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')


plt.ylabel('True Positive Rate(sensitivity)')
plt.xlabel('False Positive Rate (1-specificity)')
plt.show()#ROC curve



b=model.feature_importances_#the improtance of each variable in the dataset
print(b)
#ST_Slpo_Llat is the most improtant feature with a improtance of 0.4227

tree.plot_tree(model)



confmat=confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(confmat, cmap='GnBu', alpha=0.75)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center', size='large') 
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Confusion Matrix')
plt.show()#confusion matric


#random forest
forest=RandomForestClassifier(n_estimators=10, n_jobs=2)
forest.fit(x_train, y_train)
forest_score=accuracy_score(y_pred, y_test)
print(forest_score)


y_pred_forest = forest.predict(x_test)
confmat_forest=confusion_matrix(y_true=y_test, y_pred=y_pred_forest)
print(confmat_forest)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(confmat_forest, cmap='GnBu', alpha=0.75)
for i in range(confmat_forest.shape[0]):
    for j in range(confmat_forest.shape[1]):
        ax.text(x=j, y=i, s=confmat_forest[i, j], va='center', ha='center')
        
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Confusion Matrix')
plt.show()

y_pred_proba_forest=forest.predict_proba(x_test)
auc_score_forest=roc_auc_score(y_test, y_pred_proba_forest[:,1])
print(auc_score_forest)
#get around 0.91, which is much larger than before

fpr, tpr, thres=roc_curve(y_test, y_pred_proba_forest[:,1])
plt.plot(fpr,tpr, lw=1)
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.ylabel('True Positive Rate(sensitivity)')
plt.xlabel('False Positive Rate (1-specificity)')
plt.show()#ROC curve



