import pandas as pd
from pandas.core.frame import DataFrame 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab  
import seaborn as sns 
import easygui as gui
import joblib
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
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
# from sklearn.pipeline import Pipeline
# from sklearn.manifold import TSNE 
# from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

def read_data(df):
    '''
    This function read a dataframe and return a new datafram with only numbers 

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


def plot_confusion_matrix(y_test, y_pred,modelname):
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(conf_matrix, cmap='GnBu', alpha=0.75)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large') 
    plt.xlabel('Predictions', fontsize=10)
    plt.ylabel('Actuals', fontsize=10)
    plt.title('Confusion Matrix: %s '% modelname, fontsize=12)
    plt.show()
    fig.savefig("%s_confusion_matrix.png" % modelname)
    return None


def ROC_curve(y_test, y_pred_proba,modelname):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
    roc_auc = auc(fpr, tpr)
    fig = plt.plot(fpr, tpr, 'b', label = '%s (AUC = %0.2f)' % (modelname,roc_auc))
    plt.plot([0, 1], [0, 1],'r--', label='No Skill Classifier')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve: %s '% modelname)
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlabel('False Positive Rate (1-specificity)')
    plt.show()
    fig.savefig("%s_ROC_curve.png" % modelname)
    

def PR_curve(y_test, y_pred_proba,modelname):
    precision, recall, _= precision_recall_curve(y_test,y_pred_proba[:,1])
    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig = plt.plot(recall, precision, marker='.', label='%s ' % modelname)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve: %s '% modelname)
    plt.legend()
    plt.show()
    fig.savefig("%s_PR_curve.png" % modelname)


def Scores(y_test,y_pred,y_pred_proba):
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred_proba[:,1]))
    return None
    

def logisticRegression(dataset):
    X = dataset.drop(['HeartDisease'], axis=1)
    Y = dataset['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    lr=LogisticRegression(class_weight="balanced")
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    y_pred_proba=lr.predict_proba(x_test)
    joblib.dump(lr, "lr_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Logistic Regression')
    ROC_curve(y_test, y_pred_proba,'Logistic Regression')
    PR_curve(y_test, y_pred_proba,'Logistic Regression')
    Scores(y_test,y_pred,y_pred_proba)
    data=read_data()
    plot_learning_curve(data,lr)
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
    y_pred=rf.predict(x_test)
    y_pred_proba=rf.predict_proba(x_test)
    joblib.dump(rf, "rf_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'RandomForest')
    ROC_curve(y_test, y_pred_proba,'RandomForest')
    PR_curve(y_test, y_pred_proba,'RandomForest')
    Scores(y_test,y_pred,y_pred_proba)
    data=read_data()
    plot_learning_curve(data,rf)
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
    joblib.dump(model, "dt_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Decision Tree')
    ROC_curve(y_test, y_pred_proba,'Decision Tree')
    PR_curve(y_test, y_pred_proba,'Decision Tree')
    Scores(y_test,y_pred,y_pred_proba)
    data=read_data()
    plot_learning_curve(data,model)
    return y_test, y_pred, y_pred_proba



def gaussian_nb(dataset):
    x=dataset.drop(columns='HeartDisease')#dataset except target
    y=dataset['HeartDisease']
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)#data split
    GNB=GaussianNB()
    GNB.fit(x_train, y_train)#training
    y_pred=GNB.predict(x_test)
    y_pred_proba=GNB.predict_proba(x_test)
    joblib.dump(GNB, "gnb_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Gaussian Naive Bayes')
    ROC_curve(y_test, y_pred_proba,'Gaussian Naive Bayes')
    PR_curve(y_test, y_pred_proba,'Gaussian Naive Bayes')
    Scores(y_test,y_pred,y_pred_proba)
    data=read_data()
    plot_learning_curve(data,GNB)
    return y_test, y_pred, y_pred_proba


def plot_learning_curve(dataset, estimator):
    x=dataset.drop(columns='HeartDisease')#dataset except target
    y=dataset['HeartDisease']#target
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve",fontsize='small')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b") 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.legend(loc="best")

    plt.show()


def gui_visual():
    end = []
    while len(end) == 0: 
        gui.msgbox('Please Input patients information')
        # features = ['Age','Sex','ChestPainType(TA,ATA,ASY,NAP)','RestingBP','Cholesterol','FastingBS','RestingECG(Normal,ST,LVH)','MaxHR','ExerciseAngina','Oldpeak','ST_Slope(Up,Flat,Down)']
        # patience_information = gui.multenterbox(msg=' Please input patient information', title=' Heart Failure Prediction', fields=features, values=[])
        model_choice = gui.choicebox(msg='Which models would you like to use ', title=' Heart Failure Prediction', choices=['Logistic Regression','Random Forest','Decision Tree','Gaussian Naive Bayes'])
        gui.msgbox('Click to see your predicted result')
        # print(patience_information)
        # print(model_choice)
        patience_information_p = ['40', 'M', 'ATA', '140', '289', '0', 'Normal', '172', 'N', '0', 'Up']
        # patience_information_n = ['49', 'F', 'NAP', '160', '180', '0', 'Normal', '156', 'N', '1', 'Flat']
        new_input = DataFrame(patience_information_p).T
        new_input.columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
        new_input.insert(new_input.shape[1],'HeartDisease','0')
        new_input.apply(pd.to_numeric, errors='ignore')
        new_input.info()
        new_input[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']] = new_input[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']].apply(pd.to_numeric)
        
        if model_choice == 'Logistic Regression':
            new_model = joblib.load("lr_model.joblib")
            new_pred_data = read_data(new_input)
            x_new = new_pred_data.drop(['HeartDisease'], axis=1)
            result = new_model.predict(x_new)
            prob = new_model.predict_proba(x_new)
            print(prob)
            print(result)
            name = prob_draw(prob[0][1],prob[0][0],model_choice)
            image = name
            msg = "This is the prediction of heart failure based on %s model" % model_choice
            choices = ["Believe","Retest"]
            cho =gui.buttonbox(msg, image=image, choices = choices)
            if cho == 'Believe':
                end.append(cho)
            break
        elif model_choice =='Random Forest':
            new_model = joblib.load("rf_model.joblib")
            new_pred_data = read_data(new_input)
            x_new = new_pred_data.drop(['HeartDisease'], axis=1)
            result = new_model.predict(x_new)
            prob = new_model.predict_proba(x_new)
            name = prob_draw(prob[0][1],prob[0][0],model_choice)
            image = name
            msg = "This is the prediction of heart failure based on %s model" % model_choice
            choices = ["Belive","Retest"]
            cho =gui.buttonbox(msg, image=image, choices = choices)
            if cho == 'Believe':
                end.append(cho)
            break
        elif model_choice =='Decision Tree':
            new_model = joblib.load("dt_model.joblib")
            new_pred_data = read_data(new_input)
            x_new = new_pred_data.drop(['HeartDisease'], axis=1)
            result = new_model.predict(x_new)
            prob = new_model.predict_proba(x_new)
            name = prob_draw(prob[0][1],prob[0][0],model_choice)
            image = name
            msg = "This is the prediction of heart failure based on %s model" % model_choice
            choices = ["Belive","Retest"]
            cho =gui.buttonbox(msg, image=image, choices = choices)
            if cho == 'Believe':
                end.append(cho)
            break
        elif model_choice == 'Gaussian Naive Bayes':
            new_model = joblib.load("gnb_model.joblib")
            new_pred_data = read_data(new_input)
            x_new = new_pred_data.drop(['HeartDisease'], axis=1)
            result = new_model.predict(x_new)
            prob = new_model.predict_proba(x_new)
            name = prob_draw(prob[0][1],prob[0][0],model_choice)
            image = name
            msg = "This is the prediction of heart failure based on %s model" % model_choice
            choices = ["Belive","Retest"]
            cho =gui.buttonbox(msg, image=image, choices = choices)
            if cho == 'Believe':
                end.append(cho)
            break
    

def prob_draw(positive,negative,fptr):
    labels=['Predicted to have heart failure','Predicted healthy']
    X=[positive,negative]  
    colors = ['firebrick', 'olive']
    fig = plt.figure(figsize=(6, 3))
    plt.pie(X,labels=labels,autopct='%1.2f%%',colors = colors) #画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.title("Pie chart of %s" % fptr)

    fig.savefig("%s_PieChart.png" % fptr)
    return "%s_PieChart.png" % fptr


if __name__ == '__main__':
    df = pd.read_csv('heart.csv')
    data = read_data(df)
    logisticRegression(data)
    RandomForest(data)
    decision_tree(data)
    gaussian_nb(data)
    EDA(data)
    gui_visual()
