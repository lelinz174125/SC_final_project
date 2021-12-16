import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib
from PIL import ImageTk
from PIL import Image as PILImage
from tkinter import * 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.manifold import TSNE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pandas.plotting import parallel_coordinates
import unittest 
from unittest import TestCase


def read_data(df):
    '''
    This function read a dataframe and return a new datafram with only numbers 

    **Parameters**
        df: *dataframe*
            The original dataframe from the open database

    **Return**
        new_df: *dataframe*
            Transform all the data into number
    '''
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
    # address the data, transfer all the characters into number 
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
    # Store the new data set into a new file
    new_df.to_csv('heart_data_addressed.csv')
    return(new_df)


def corr(dataset):
    '''
    This function read a data set and figure out the potential relationship in data set 

    **Parameters**
        dataset: *dataframe*
            The roughly addressed dataframe

    **Return**
        None
    '''
    # find the corresponding between features and outcome
    corr = dataset.corr(method = 'spearman')
    fig = plt.figure(figsize=(20,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Spearman Correlation Heatmap')
    plt.show()
    fig.savefig('Figure/feature_exploration.png')


def data_clean(dataset):
    '''
    This function read a dataframe and return a new datafram with only important features

    **Parameters**
        dataset: *dataframe*
            The roughly addressed dataframe

    **Return**
        new_df: *dataframe*
            A new data frame that exclude the less relative columns
    '''
    # drop the unrelative data set 
    new_df = dataset.drop(['Age','Sex','ChestPainType_TA','ChestPainType_NAP','RestingBP','Cholesterol','FastingBS',
                            'RestingECG_Normal','RestingECG_ST','RestingECG_LVH','ST_Slope_Down'], axis=1)
    new_df.head(10)
    new_df.info()
    return new_df


def kmeans_find_cluster(dataset):
    '''
    This function find the optimal number of cluster.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    X=dataset.drop(columns='HeartDisease')
    #Dataset except target
    n_clusters = [2,3,4,5,6,7,8] 
    #Number of clusters
    meandistortions = []
    for n in n_clusters:
        kmeans= KMeans(n_clusters=n, init='k-means++')
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])
    #Calculate the Euclidean distance between each cluster centers
    fig, ax = plt.subplots(figsize=(12,5))
    ax = sns.lineplot(n_clusters, meandistortions, marker='o', ax=ax)
    #Plot elbow method with the number of clusters and Euclidean distance
    ax.set_title("Elbow method")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("average dispersion")
    ax.axvline(3, ls="--", c="red")
    plt.grid()
    plt.show()
    fig.savefig("Figure/%s_k_means.png" %'clustering compare')
    
    
def para_coor(dataset):
    '''
    This function plot the parallel coordinates for each parameters.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    X=dataset.drop(columns='HeartDisease')
    #Dataset except target
    plt.rcParams['figure.figsize'] = (30.0, 12.0)
    parallel_coordinates(X,'Sex')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=5,fancybox=True,shadow=True)
    plt.title('Parallel Coordination (Sex)', fontsize=12)
    plt.savefig('Figure/Parellel_Coordination_Sex.png')
    plt.show()
    #show parallel coordinates between sex and remaining parameters.
    parallel_coordinates(X,'Age')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=30,fancybox=True,shadow=True)
    plt.title('Parallel Coordination (Age)', fontsize=12)
    plt.savefig('Figure/Parellel_Coordination_Age.png')
    plt.show()
    #show parallel coordinates between age and remaining parameters.
    parallel_coordinates(X,'RestingBP')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=30,fancybox=True,shadow=True)
    plt.title('Parallel Coordination (RestingBP)', fontsize=12)
    plt.savefig('Figure/Parellel_Coordination_RestingBP.png')
    plt.show()
    #show parallel coordinates between RestingBP and remaining parameters.
    parallel_coordinates(X,'Cholesterol')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=30,fancybox=True,shadow=True)
    plt.title('Parallel Coordination (Cholesterol)', fontsize=12)
    plt.savefig('Figure/Parellel_Coordination_Cholesterol.png')
    plt.show()
    #show parallel coordinates between Cholesterol and remaining parameters.
    parallel_coordinates(X,'MaxHR')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=30,fancybox=True,shadow=True)
    plt.title('Parallel Coordination (MaxHR)', fontsize=12)
    plt.savefig('Figure/Parellel_Coordination_MaxHR.png')    
    plt.show()
    #show parallel coordinates between MaxHR and remaining parameters.
    parallel_coordinates(X,'Oldpeak')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=30,fancybox=True,shadow=True)
    plt.title('Parallel Coordination (Oldpeak)', fontsize=12)
    plt.savefig('Figure/Parellel_Coordination_Oldpeak.png')    
    plt.show()
    #show parallel coordinates between Oldpeak and remaining parameters.
    #plots of remaining parameters are not shown in the coding above since their plots are the same as others


def t_SNE(dataset):
    '''
    This function read a dataset and visualize the high dimensional data through 
    reduce it into two dimension

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    # drop the outcome
    y_true = dataset['HeartDisease']
    x_data = dataset.drop(columns='HeartDisease')
    # use t-SNE to reduce the dimentionality of dataset
    tsne = TSNE(verbose = 0, n_components=2, init='random', perplexity = 50 , n_iter=1000, learning_rate=10)
    X_tsne = tsne.fit_transform(x_data) 
    X_tsne_data = np.vstack((X_tsne.T, y_true)).T 
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2','class']) 
    df_tsne.head()
    fig = plt.figure(figsize=(8, 8)) 
    plt.title('The t-SNE analysis')
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2',s=4) 
    fig.savefig('Figure/t-SNE_analysis.png')


def plot_confusion_matrix(y_test, y_pred,modelname):
    '''
    This function read the true outcome and the predicted outcome,
    and draw a confusion matrix of specific model

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred: *array*
            The predicted outcome
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    # calculate the confusion matrix based on different model 
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(conf_matrix, cmap='GnBu', alpha=0.75)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large') 
    plt.xlabel('Predictions', fontsize=10)
    plt.ylabel('Actuals', fontsize=10)
    plt.title('Confusion Matrix: %s '% modelname, fontsize=12)
    fig.savefig('Figure/%s_confusion_matrix.png' % modelname)


def ROC_curve(y_test, y_pred_proba,modelname):
    '''
    This function read the true outcome and the probablity of the positive predicted outcomes,
    and draw a ROC curve of specific model

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred_proba: *array*
            The probablity of the positive predicted outcomes
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    # calculate the false positive rate and true positive rate of model
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
    # calculate the auc value of model
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, 'b', label = '%s (AUC = %0.2f)' % (modelname,roc_auc))
    #plot the ROC curve and calculate AUC
    plt.plot([0, 1], [0, 1],'r--', label='No Skill Classifier')
    #plot the 'No Skill Classifier' curve
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
    #Plot the 'perfect performance' curve
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve: %s '% modelname)
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlabel('False Positive Rate (1-specificity)')
    fig.savefig("Figure/%s_ROC_curve.png" % modelname)
    

def PR_curve(y_test, y_pred_proba,modelname):
    '''
    This function read the true outcome and the probablity of the positive predicted outcomes,
    and draw a ROC curve of specific model

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred_proba: *array*
            The probablity of the positive predicted outcomes
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    precision, recall, _= precision_recall_curve(y_test,y_pred_proba[:,1])
    #calculate precision and recall
    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig = plt.figure()
    plt.plot(recall, precision, marker='.', label='%s ' % modelname)]
    #plot PR curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill Classifier')
    #plot 'No Skill Classifier' curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve: %s '% modelname)
    plt.legend()
    fig.savefig("Figure/%s_PR_curve.png" % modelname)


def Scores(y_test,y_pred,y_pred_proba,modelname):
    '''
    This function read the true outcome, the predicted outcome 
    and the probablity of the positive predicted outcomes. 
    Then, it give back the precison, recall, F1 score and the ROC-AUC score of the model.

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred: *array*
            The predicted outcomes
        y_pred_proba: *array*
            The probablity of the positive predicted outcomes
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    print('This is %s'% modelname)
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    #Calculate the scores by test set and probability estimates provided by the predict_pred
    print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred_proba[:,1]))
    #Calculate the ROC-AUC score by test set result probability estimates provided by the predict_proba
    return None
    

def plot_learning_curve(dataset, estimator,modelname):
    '''
    This function read the dataset and give back its learning curve based on specific model.

    **Parameters**
        dataset: *dataframe*
            The dataframe used for model fitting
        estimator: *array*
            The fitting model
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    x=dataset.drop(columns='HeartDisease')
    #Dataset except target
    y=dataset['HeartDisease']
    #Dataset of target
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y)
    #Number of samples in training set, score if training set, score of test set
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #Mean of test set scores
    test_scores_std = np.std(test_scores, axis=1)
    #Standard deviation of test scores
    fig = plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title('Learning Curve: %s '% modelname, fontsize='small')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b") 
    #Plot the learning curve with upper and lower limits of training score
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
    #Plot the learning curve with upper and lower limits of test score
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    #Plot learning curve
    plt.legend(loc="best")
    fig.savefig("Figure/%s_Learning_curve.png" % modelname)


def logisticRegression(dataset):
    '''
    This function read a dataset and train it on logistic regression model.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    X = dataset.drop(['HeartDisease'], axis=1)
    #Dataset except target
    Y = dataset['HeartDisease']
    #Dataset of target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
    #Split data to training set and test set
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #Feature scaling
    lr=LogisticRegression(class_weight="balanced")
    lr.fit(x_train,y_train)
    #Fit the model to training set
    y_pred=lr.predict(x_test)
    #Predict the result of test set
    y_pred_proba=lr.predict_proba(x_test)
    #Predict the result of test set to plot ROC and calculate AUC
    joblib.dump(lr, "Model/lr_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Logistic Regression')
    ROC_curve(y_test, y_pred_proba,'Logistic Regression')
    PR_curve(y_test, y_pred_proba,'Logistic Regression')
    plot_learning_curve(dataset,lr,'Logistic Regression')
    Scores(y_test,y_pred,y_pred_proba,'Logistic Regression')
    return y_test, y_pred, y_pred_proba


def RandomForest(dataset):
    '''
    This function read a dataset and train it on random forest model.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    X = dataset.drop(['HeartDisease'], axis=1)
    #Dataset except target
    Y = dataset['HeartDisease']
    #Dataset of target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
    #Split data to training set and test set
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #Feature scaling
    rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    rf.fit(x_train,y_train)
    #Fit the model to training set
    y_pred=rf.predict(x_test)
    #Predict the result of test set
    y_pred_proba=rf.predict_proba(x_test)
    #Predict the result of test set to plot ROC and calculate AUC
    joblib.dump(rf, "Model/rf_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'RandomForest')
    ROC_curve(y_test, y_pred_proba,'RandomForest')
    PR_curve(y_test, y_pred_proba,'RandomForest')
    plot_learning_curve(dataset,rf,'RandomForest')
    Scores(y_test,y_pred,y_pred_proba,'RandomForest')
    return y_test, y_pred, y_pred_proba


def decision_tree(dataset):
    '''
    This function read a dataset and train it on decision tree model.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    x=dataset.drop(columns='HeartDisease')
    #Dataset except target
    y=dataset['HeartDisease']
    #Dataset of target
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)
    #Split data to training set and test set
    model= DecisionTreeClassifier(criterion='gini',
                                       max_depth=6,
                                       max_leaf_nodes=14,
                                       min_samples_leaf=1)
    model.fit(x_train, y_train)
    #Fit the model with optimal parameters to training set
    y_pred=model.predict(x_test)
    #Predict the result of test set
    y_pred_proba=model.predict_proba(x_test)
    #Predict the result of test set to plot ROC and calculate AUC
    tree.plot_tree(model)
    #Virtualise decision tree
    joblib.dump(model, "Model/dt_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Decision Tree')
    ROC_curve(y_test, y_pred_proba,'Decision Tree')
    PR_curve(y_test, y_pred_proba,'Decision Tree')
    plot_learning_curve(dataset,model,'Decision Tree')
    Scores(y_test,y_pred,y_pred_proba,'Decision Tree')
    return y_test, y_pred, y_pred_proba


def gaussian_nb(dataset):
    '''
    This function read a dataset and train it on gaussian naive bayes model.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    x=dataset.drop(columns='HeartDisease')
    #Dataset except target
    y=dataset['HeartDisease']
    #Dataset of target
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)
    #Split data to training set and test set
    GNB=GaussianNB()
    GNB.fit(x_train, y_train)
    #Fit the model to training set
    y_pred=GNB.predict(x_test)
    #Predict the result of test set
    y_pred_proba=GNB.predict_proba(x_test)
    #Predict the result of test set to plot ROC and calculate AUC
    joblib.dump(GNB, "Model/gnb_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Gaussian Naive Bayes')
    ROC_curve(y_test, y_pred_proba,'Gaussian Naive Bayes')
    PR_curve(y_test, y_pred_proba,'Gaussian Naive Bayes')
    plot_learning_curve(dataset,GNB,'Gaussian Naive Bayes')
    Scores(y_test,y_pred,y_pred_proba,'Gaussian Naive Bayes')
    return y_test, y_pred, y_pred_proba

  
def input_gui():
    '''
    This function constructure a GUI interface 
    and allow people input a new patient value through the interface.
    The information of new patient will be returned.

    **Parameters**
         None

    **Return**
        pada: *dataframe*
            The information of new patient
    '''
    pada = pd.DataFrame(np.array([['40', 'M', 'ATA', '140', '289', '0', 'Normal', '172', 'N', '0', 'Up']])
                        ,columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                        'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'])
    def clicked():
        '''
        After clicked submit point.
        This function collect all the input value of a new patient into pada dataframe.

        **Parameters**
            None

        **Return**
            pada: *dataframe*
                The information of new patient
        '''
        pada['Age']=age.get()
        pada['Sex']=sex.get()
        pada['ChestPainType']=ChestPainType.get()
        pada['RestingBP'] = RestingBP.get()
        pada['Cholesterol']=Cholesterol.get()
        pada['FastingBS']=FastingBS.get()
        pada['RestingECG']=RestingECG.get()
        pada['MaxHR']=MaxHR.get()
        pada['ExerciseAngina']=ExerciseAngina.get()
        pada['Oldpeak']=Oldpeak.get()
        pada['ST_Slope']=ST_Slope.get()
        pada['Sex'].replace([1,2],['M', 'F'], inplace = True)
        pada['ChestPainType'].replace([1,2,3,4],['TA','ATA','ASY','NAP'], inplace = True)
        pada['RestingECG'].replace([1,2,3],['Normal','ST','LVH'], inplace = True)
        pada['ExerciseAngina'].replace([1,2], ['Y','N'],inplace = True)
        pada['ST_Slope'].replace([1,2,3], ['Up','Flat','Down'],inplace = True)
        pada.to_csv('new_patient_data.csv')
        frame1.quit()

    toop = Tk()
    toop.title("Heart Failure Prediction")
    Label(toop,text="Please input patient information").pack()
    frame1 = Frame(toop)

    age = StringVar()
    Label(frame1,text="Age").grid(row=0,column=0,sticky=W)
    Entry(frame1,text="Input",textvariable=age).grid(row=1,column=0)
  

    Label(frame1,text="Sex:").grid(row=2,column=0,sticky=W)
    list1 = [("M", 1),("F", 2)]
    sex = IntVar()
    i = 3
    for num1, check1 in list1:
        Radiobutton(frame1, text=num1, variable=sex,value=check1).grid(row=i, column=0, sticky=W)
        i += 1

    Label(frame1,text="ChestPainType:").grid(row=i+1,column=0,sticky=W)
    list2 = [("TA", 1),("ATA", 2),("ASY", 3),("NAP", 4)]
    ChestPainType = IntVar()
    k = i+3
    for num2, check2 in list2:
        Radiobutton(frame1, text=num2, variable=ChestPainType,value=check2).grid(row=k, column=0, sticky=W)
        k += 1

    RestingBP = StringVar()
    Label(frame1,text="RestingBP:").grid(row=k+1,column=0,sticky=W)
    Entry(frame1,text="Input",textvariable=RestingBP).grid(row=k+2,column=0)

    Cholesterol = StringVar()
    Label(frame1,text="Cholesterol:").grid(row=k+3,column=0,sticky=W)
    Entry(frame1,text="Input",textvariable=Cholesterol ).grid(row=k+4,column=0)

    FastingBS = StringVar()
    Label(frame1,text="FastingBS:").grid(row=k+5,column=0,sticky=W)
    Entry(frame1,text="Input",textvariable=FastingBS).grid(row=k+6,column=0)

    Label(frame1,text="RestingECG:").grid(row=k+7,column=0,sticky=W)
    list3 = [("Normal", 1),("ST", 2),("LVH", 3),]
    RestingECG = IntVar()
    a = k+8
    for num3, check3 in list3:
        Radiobutton(frame1, text=num3, variable=RestingECG,value=check3).grid(row=a, column=0, sticky=W)
        a += 1

    MaxHR = StringVar()
    Label(frame1,text="MaxHR").grid(row=a+1,column=0,sticky=W)
    Entry(frame1,text="Input",textvariable=MaxHR).grid(row=a+2,column=0)

    Label(frame1,text="ExerciseAngina").grid(row=a+3,column=0,sticky=W)
    list4 = [("Yes", 1),("No", 2)]
    ExerciseAngina = IntVar()
    b = a+4
    for num4, check4 in list4:
        Radiobutton(frame1, text=num4, variable=ExerciseAngina,value=check4).grid(row=b, column=0, sticky=W)
        b += 1

    Oldpeak = StringVar()
    Label(frame1,text="Oldpeak").grid(row=b+1,column=0,sticky=W)
    Entry(frame1,text="Input",textvariable=Oldpeak).grid(row=b+2,column=0)

    Label(frame1,text="ST_Slope").grid(row=b+3,column=0,sticky=W)
    list4 = [("Up", 1),("Flat", 2),('Down',3)]
    ST_Slope = IntVar()
    c = b+4
    for num4, check4 in list4:
        Radiobutton(frame1, text=num4, variable=ST_Slope,value=check4).grid(row=c, column=0, sticky=W)
        c += 1

    frame1.pack(padx=20,pady=20)
    Button(toop,text="Submit",bg="white",fg="blue",command=clicked).pack(side=BOTTOM)
    mainloop()
    return pada


def visual_gui(new_input):
    '''
    This function constructure a GUI interface and allow people to choose a ML model

    **Parameters**
        new_input: *dataframe*
            The information of new patient

    **Return**
        None
    '''
    new_input.insert(new_input.shape[1],'HeartDisease','0')
    new_input.apply(pd.to_numeric, errors='ignore')
    new_input.info()
    def clicked():
        model_choice = choice.get()
        if model_choice == 1 :
            model = "Logistic Regression"
            new_model = joblib.load("Model/lr_model.joblib")
            show_gui(new_input,new_model,model)
        elif model_choice ==2 :
            model = "Random Forest"
            new_model = joblib.load("Model/rf_model.joblib")
            show_gui(new_input,new_model,model)
        elif model_choice ==3 :
            model = 'Decision Tree'
            new_model = joblib.load("Model/dt_model.joblib")
            show_gui(new_input,new_model,model)
        elif model_choice == 4 :
            model = 'Gaussian Naive Bayes'
            new_model = joblib.load("Model/gnb_model.joblib")
            show_gui(new_input,new_model,model)
        frame2.quit()

    frame2 = Toplevel()
    Label(frame2,text="Which models would you like to use for heart failure prediction?").grid(row=0,column=0,sticky=W)
    list = [("Logistic Regression", 1),("Random Forest", 2),('Decision Tree',3),('Gaussian Naive Bayes',4)]
    choice = IntVar()
    c = 1
    for num, check in list:
        Radiobutton(frame2, text=num, variable=choice,value=check).grid(row=c, column=0, sticky=W)
        c += 1
    Button(frame2,text="Predict",bg="white",fg="blue",command=clicked).grid(row=c+1, column=1, sticky=W)
    mainloop()


def show_gui(new_input,new_model,model_choice):
    '''
    This function constructure a GUI interface and exhibit the predcition results

    **Parameters**
         new_input: *dataframe*
            The information of new patient
         new_model: *dataframe*
            The model we want to used for prediction 
         model_choice: *dataframe*
            The name of chosen model 

    **Return**
        None
    '''
    new_pred_data = data_clean(read_data(new_input))
    x_new = new_pred_data.drop(['HeartDisease'], axis=1)
    prob = new_model.predict_proba(x_new)
    name = prob_draw(prob[0][1],prob[0][0],model_choice)
    frame3 = Toplevel()
    im=PILImage.open(name)
    img=ImageTk.PhotoImage(im)
    Label(frame3,image=img).pack()
    Button(frame3,text="Quit",command=quit).pack(side=BOTTOM)
    mainloop()
    

def prob_draw(positive,negative,fptr):
    '''
    This function draw the pie chart of prediction results 

    **Parameters**
        positive: *int*
            The predicted probability of heart failure
        negative: *int*
            The predicted probability of being healthy
        fptr: *str*
            The name of the model
         
    **Return**
        The name of the pie chart figure.
    '''
    labels=['Predicted to have heart failure(Red color)','Predicted healthy(Green color)']
    X=[positive,negative]  
    colors = ['firebrick', 'olive']
    fig = plt.figure(figsize=(8, 4))
    plt.pie(X,labels=labels,autopct='%1.2f%%',colors = colors) 
    plt.title("Predicted results: %s" % fptr)
    fig.savefig("Figure/%s_PieChart.png" % fptr)
    return "Figure/%s_PieChart.png" % fptr

    
class test(unittest.TestCase):
    '''
    This class can test if the read_data, data_clean and input_gui function can oprate correctly.
    '''
    
    def setUp(self):
        '''
        This function will set up all parameters.

        Returns
        -------
        None.

        '''
        self.data=read_data(df)
        self.cleaned_data=data_clean(data)
        self.in_gui=input_gui()
        

    def test_read(self):
        '''
        This function tests if the read_data function can works correctly.

        Returns
        -------
        None.

        '''        
        result1=self.data['Age'][0]
        result2=self.data['Sex'][0]
        num_col1=len(self.data.columns)
        self.assertEqual(result1,40)
        self.assertEqual(result2,1.0)
        self.assertEqual(num_col1,19)
        #if the first value of age is '40' 
        #the first value of sex is '0' 
        #the number of columns in data is '10'
        #the read_data function works correctly
        
    def test_dataclean(self):
        '''
        This function tests data_clean function.

        Returns
        -------
        None.

        '''
        num_col2=len(self.cleaned_data.columns)
        self.assertEqual(num_col2,8)
        #if the number of columns in cleaned_data is '8'
        #the data_clean function works correctly
        
    def test_inputgui(self):
        '''
        This function tests if the input_gui function works correctly.

        Returns
        -------
        None.

        '''
        num_col3=len(self.in_gui.columns)
        self.assertEqual(num_col3,11)
        #if the number of columns of in_gui (dataset patient input ) is '11'
        #the input_gui function works correctly
    

if __name__ == '__main__':
    df = pd.read_csv('heart.csv')
    data = read_data(df)
    # corr(data)
    # kmeans_find_cluster(data)
    # para_coor(data)
    # t_SNE(data)
    cleaned_data = data_clean(data)
    # logisticRegression(cleaned_data)
    # RandomForest(cleaned_data)
    # decision_tree(cleaned_data)
    # gaussian_nb(cleaned_data)
    new_patient_info = input_gui()
    visual_gui(new_patient_info)
    unittest.main()
