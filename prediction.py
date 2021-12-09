#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:13:46 2021

@author: xihongshijidanzhajiangmian
"""

import pandas as pd
from pandas.core.frame import DataFrame 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import easygui as gui
import joblib
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler 
# from sklearn.manifold import TSNE 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_recall_curve,precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.metrics import confusion_matrix


def read_data():
    '''
    This function read a dataframe and return a new datafram with only numbers 

    '''

    df = pd.read_csv('heart.csv')
    '''
    The features in the data set incluses
        Age: age of the patient [years]
        Sex: sex of the patient [1: Male, 0: Female]
        ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
        RestingBP: resting blood pressure [mm Hg]
        Cholesterol: serum cholesterol [mm/dl]
        FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
        RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality 
                    (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable 
                    or definite left ventricular hypertrophy by Estes' criteria]
        MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
        ExerciseAngina: exercise-induced angina [1: Yes, 0: No]
        Oldpeak: oldpeak = ST [Numeric value measured in depression]
        ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
        HeartDisease: output class [1: heart disease, 0: Normal]
    '''
    df.info()
    df.head(10)
    new_df = pd.DataFrame(columns=['Age','Sex',	'ChestPainType_TA','ChestPainType_ATA','ChestPainType_ASY',
                                'ChestPainType_NAP','RestingBP','Cholesterol','FastingBS','RestingECG_Normal',
                                'RestingECG_ST','RestingECG_LVH','MaxHR','ExerciseAngina','Oldpeak','ST_Slope_Up',
                                'ST_Slope_Flat','ST_Slope_Down','HeartDisease'])
    new_df['Age']=df['Age']
    new_df['Sex']=df['Sex']
    new_df['ChestPainType_TA']=df['ChestPainType']
    new_df['ChestPainType_ATA']=df['ChestPainType']
    new_df['ChestPainType_ASY']=df['ChestPainType']
    new_df['ChestPainType_NAP']=df['ChestPainType']
    new_df['RestingBP']=df['RestingBP']
    new_df['Cholesterol']=df['Cholesterol']
    new_df['FastingBS']=df['FastingBS']
    new_df['RestingECG_Normal']=df['RestingECG']
    new_df['RestingECG_ST']=df['RestingECG']
    new_df['RestingECG_LVH']=df['RestingECG']
    new_df['MaxHR']=df['MaxHR']
    new_df['ExerciseAngina']=df['ExerciseAngina']
    new_df['Oldpeak']=df['Oldpeak']
    new_df['ST_Slope_Up']=df['ST_Slope']
    new_df['ST_Slope_Flat']=df['ST_Slope']
    new_df['ST_Slope_Down']=df['ST_Slope']
    new_df['HeartDisease']=df['HeartDisease']

    new_df['Sex'].replace(['M','F'],[1.0, 0.0], inplace = True)
    new_df['ChestPainType_TA'].replace(['TA','ATA','ASY','NAP'],[1.0, 0.0, 0.0, 0.0], inplace = True)
    new_df['ChestPainType_ATA'].replace(['TA','ATA','ASY','NAP'],[0.0, 1.0, 0.0, 0.0], inplace = True)
    new_df['ChestPainType_ASY'].replace(['TA','ATA','ASY','NAP'],[0.0, 0.0, 0.1, 0.0], inplace = True)
    new_df['ChestPainType_NAP'].replace(['TA','ATA','ASY','NAP'],[0.0, 0.0, 0.0, 1.0], inplace = True)
    new_df['RestingECG_Normal'].replace(['Normal','ST','LVH'],[1.0, 0.0, 0.0], inplace = True)
    new_df['RestingECG_ST'].replace(['Normal','ST','LVH'],[0.0, 1.0, 0.0], inplace = True)
    new_df['RestingECG_LVH'].replace(['Normal','ST','LVH'],[0.0, 0.0, 1.0], inplace = True)
    new_df['ExerciseAngina'].replace(['Y','N'],[1.0, 0.0], inplace = True)
    new_df['ST_Slope_Up'].replace(['Up','Flat','Down'],[1.0, 0.0, 0.0], inplace = True)
    new_df['ST_Slope_Flat'].replace(['Up','Flat','Down'],[0.0, 1.0, 0.0], inplace = True)
    new_df['ST_Slope_Down'].replace(['Up','Flat','Down'],[0.0, 0.0, 1.0], inplace = True)

    # new_df.info()
    # new_df.head(10)
    new_df.to_csv('heart_data_addressed.csv')
    return(new_df)


def EDA(dataset):
    
    corr = dataset.corr(method = 'spearman')
    plt.figure(figsize=(20,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Spearman Correlation Heatmap')
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap='GnBu', alpha=0.75)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large') 
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('Actuals', fontsize=14)
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()
    return None



def ROC_curve(y_test, y_pred_proba):
    fpr, tpr, threholds = metrics.roc_curve(y_test, y_pred_proba[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    print(fpr,tpr,threholds)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--', label='random guessing')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlabel('False Positive Rate (1-specificity')
    plt.show()
    


def Scores(y_test,y_pred):
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    return None
    

def rocauc_score(y_test, y_pred_proba):
    print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred_proba[:,1]))
    return None


def PR_curve(y_test, y_pred):
    precision, recall, _= precision_recall_curve(y_test, y_pred)
    print(precision,recall)
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparison of classification models (PRC)')
    plt.legend()
    plt.show()


def logisticRegression(dataset):
    X = dataset.drop(['HeartDisease'], axis=1)
    Y = dataset['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    lr=LogisticRegression(class_weight="balanced")
    lr.fit(x_train,y_train)
    #利用训练模型进行预测
    y_pred=lr.predict(x_test)
    # res2 = lr.fit()
    # res2.summary()
    y_pred_proba=lr.predict_proba(x_test)
    joblib.dump(lr, "lr_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred)
    ROC_curve(y_test, y_pred_proba)
    PR_curve(y_test, y_pred)
    Scores(y_test,y_pred)
    rocauc_score(y_test,y_pred_proba)
    return y_test, y_pred, y_pred_proba


def RandomForest(dataset):
    X = dataset.drop(['HeartDisease'], axis=1)
    Y = dataset['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    
    rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    rf.fit(x_train,y_train)
    #利用训练模型进行预测
    
    y_pred=rf.predict(x_test)
    print(y_pred)
    y_pred_proba=rf.predict_proba(x_test)
    joblib.dump(rf, "rf_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred)
    ROC_curve(y_test, y_pred_proba)
    PR_curve(y_test, y_pred)
    Scores(y_test,y_pred)
    rocauc_score(y_test, y_pred_proba)
    
    return y_test, y_pred, y_pred_proba


def decision_tree(dataset):
    x=dataset.drop(columns='HeartDisease')#dataset except target
    y=dataset['HeartDisease']
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)#data split
    model=DecisionTreeClassifier(random_state=0)
    model.fit(x_train, y_train)#training
    y_pred=model.predict(x_test)
    y_pred_proba=model.predict_proba(x_test)
    
    tree.plot_tree(model)
    
    plot_confusion_matrix(y_test,y_pred)
    ROC_curve(y_test, y_pred_proba)
    PR_curve(y_test, y_pred)
    Scores(y_test,y_pred)
    rocauc_score(y_test, y_pred_proba)
    return y_test, y_pred, y_pred_proba


def GUI():
    gui.msgbox('Thank you for this input')
    model_choice = gui.choicebox(msg='Which models would you like to use ', title=' Heart Failure Prediction', choices=['Logistic Regression','Randomforest','Desision Tree','Gaussian Naive Bayes'])
    features = ['Age','Sex','ChestPainType(TA,ATA,ASY,NAP)','RestingBP','Cholesterol','FastingBS','RestingECG(Normal,ST,LVH)','MaxHR','ExerciseAngina','Oldpeak','ST_Slope(Up,Flat,Down)']
    patience_information = gui.multenterbox(msg=' Please input patient information', title=' Heart Failure Prediction', fields=features, values=[])
    gui.msgbox('Thank you for this input')
    print(patience_information)
    print(model_choice)
    new_input = DataFrame(patience_information).T
    new_input.columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    new_input.insert(new_input.shape[1],'HeartDisease','0')
    new_input.apply(pd.to_numeric, errors='ignore')
    print(new_input)
    new_input.info()
    new_input[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']] = new_input[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']].apply(pd.to_numeric)
    print(new_input)
    if model_choice == 'Logistic Regression':
        new_model = joblib.load("lr_model.joblib")
        new_pred_data = read_data(new_input)
        x_new = new_pred_data.drop(['HeartDisease'], axis=1)
        result = new_model.predict(x_new)
        print(result)
    elif model_choice =='Randomforest':
        new_model = joblib.load("rf_model.joblib")
        new_pred_data = read_data(new_input)
        x_new = new_pred_data.drop(['HeartDisease'], axis=1)
        result = new_model.predict(x_new)
        print(result)
    # elif model_choice =='Desision Tree':
    #     new_model = joblib.load("model.joblib")
    # elif model_choice == 'Gaussian Naive Bayes':
    #     new_model = joblib.load("model.joblib")
    


if __name__ == '__main__':
    data = read_data()
    logisticRegression(data)
    RandomForest(data)
    decision_tree(data)
    EDA(data)
    
