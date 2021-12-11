import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import easygui as gui
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
from sklearn.manifold import TSNE 
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
    new_df.to_csv('heart_data_addressed.csv')
    return(new_df)


def EDA(dataset):
    
    corr = dataset.corr(method = 'spearman')
    fig = plt.figure(figsize=(20,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Spearman Correlation Heatmap')
    plt.show()
    fig.savefig('Figure/feature_exploration.png')


def data_clean(dataset):
    
    new_df = dataset.drop(['Age','Sex','ChestPainType_TA','ChestPainType_NAP','RestingBP','Cholesterol','FastingBS',
                            'RestingECG_Normal','RestingECG_ST','RestingECG_LVH','ST_Slope_Down'], axis=1)
    new_df.head(10)
    new_df.info()
    return new_df


def kmeans():
    pass


def t_SNE(dataset):
    y_true = dataset['HeartDisease']
    x_data = dataset.drop(columns='HeartDisease')
    tsne = TSNE(verbose = 1, n_components=2, init='random', perplexity = 50 , n_iter=1000, learning_rate=10)
    X_tsne = tsne.fit_transform(x_data) 
    X_tsne_data = np.vstack((X_tsne.T, y_true)).T 
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2','class']) 
    df_tsne.head()
    plt.figure(figsize=(8, 8)) 
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2',s=4) 
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
    fig.savefig('Figure/%s_confusion_matrix.png' % modelname)
    return None


def ROC_curve(y_test, y_pred_proba,modelname):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, 'b', label = '%s (AUC = %0.2f)' % (modelname,roc_auc))
    plt.plot([0, 1], [0, 1],'r--', label='No Skill Classifier')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve: %s '% modelname)
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlabel('False Positive Rate (1-specificity)')
    fig.savefig("Figure/%s_ROC_curve.png" % modelname)
    

def PR_curve(y_test, y_pred_proba,modelname):
    precision, recall, _= precision_recall_curve(y_test,y_pred_proba[:,1])
    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig = plt.figure()
    plt.plot(recall, precision, marker='.', label='%s ' % modelname)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve: %s '% modelname)
    plt.legend()
    fig.savefig("Figure/%s_PR_curve.png" % modelname)


def Scores(y_test,y_pred,y_pred_proba):
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred_proba[:,1]))
    return None
    

def plot_learning_curve(dataset, estimator,modelname):
    x=dataset.drop(columns='HeartDisease')#dataset except target
    y=dataset['HeartDisease']#target
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig = plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title('Learning Curve: %s '% modelname, fontsize='small')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b") 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.legend(loc="best")
    fig.savefig("Figure/%s_Learning_curve.png" % modelname)


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
    joblib.dump(lr, "Model/lr_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Logistic Regression')
    ROC_curve(y_test, y_pred_proba,'Logistic Regression')
    PR_curve(y_test, y_pred_proba,'Logistic Regression')
    Scores(y_test,y_pred,y_pred_proba)
    plot_learning_curve(dataset,lr,'Logistic Regression')
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
    joblib.dump(rf, "Model/rf_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'RandomForest')
    ROC_curve(y_test, y_pred_proba,'RandomForest')
    PR_curve(y_test, y_pred_proba,'RandomForest')
    Scores(y_test,y_pred,y_pred_proba)
    plot_learning_curve(dataset,rf,'RandomForest')
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
    joblib.dump(model, "Model/dt_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Decision Tree')
    ROC_curve(y_test, y_pred_proba,'Decision Tree')
    PR_curve(y_test, y_pred_proba,'Decision Tree')
    Scores(y_test,y_pred,y_pred_proba)
    plot_learning_curve(dataset,model,'Decision Tree')
    return y_test, y_pred, y_pred_proba


def gaussian_nb(dataset):
    x=dataset.drop(columns='HeartDisease')#dataset except target
    y=dataset['HeartDisease']
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=0)#data split
    GNB=GaussianNB()
    GNB.fit(x_train, y_train)#training
    y_pred=GNB.predict(x_test)
    y_pred_proba=GNB.predict_proba(x_test)
    joblib.dump(GNB, "Model/gnb_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'Gaussian Naive Bayes')
    ROC_curve(y_test, y_pred_proba,'Gaussian Naive Bayes')
    PR_curve(y_test, y_pred_proba,'Gaussian Naive Bayes')
    Scores(y_test,y_pred,y_pred_proba)
    plot_learning_curve(dataset,GNB,'Gaussian Naive Bayes')
    return y_test, y_pred, y_pred_proba

  
def input_gui():
    pada = pd.DataFrame(np.array([['40', 'M', 'ATA', '140', '289', '0', 'Normal', '172', 'N', '0', 'Up']])
                        ,columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                        'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'])
    def clicked():
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
        print(pada)
        frame1.quit()

    toop = Tk()
    toop.title("Heart Failure Prediction")
    Label(toop,text="Please input patient information").pack()
    frame1 = Frame(toop)

    age = StringVar()
    Label(frame1,text="Age").grid(row=0,column=0,sticky=W)
    Entry(frame1,text="请输入内容",textvariable=age).grid(row=1,column=0)
  

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
    Button(toop,text="Submit",bg="blue",fg="white",command=clicked).pack(side=BOTTOM)
    # Button(toop,text="Quit",bg="blue",fg="white",command=quit).pack(side=BOTTOM)
    mainloop()
    return pada


def visual_gui(new_input):
    new_input.insert(new_input.shape[1],'HeartDisease','0')
    new_input.apply(pd.to_numeric, errors='ignore')
    new_input.info()
    new_input[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']] = new_input[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']].apply(pd.to_numeric)
    model_choice = gui.choicebox(msg='Which models would you like to use ', title=' Heart Failure Prediction', choices=['Logistic Regression','Random Forest','Decision Tree','Gaussian Naive Bayes'])
    gui.msgbox('Click to see your predicted result')
    if model_choice == 'Logistic Regression':
        new_model = joblib.load("lr_model.joblib")
        show_gui(new_input,new_model,model_choice)
        
    elif model_choice =='Random Forest':
        new_model = joblib.load("rf_model.joblib")
        show_gui(new_input,new_model,model_choice)
        
    elif model_choice =='Decision Tree':
        new_model = joblib.load("dt_model.joblib")
        show_gui(new_input,new_model,model_choice)
        
    elif model_choice == 'Gaussian Naive Bayes':
        new_model = joblib.load("gnb_model.joblib")
        show_gui(new_input,new_model,model_choice)


def show_gui(new_input,new_model,model_choice):
    new_pred_data = read_data(new_input)
    x_new = new_pred_data.drop(['HeartDisease'], axis=1)
    prob = new_model.predict_proba(x_new)
    name = prob_draw(prob[0][1],prob[0][0],model_choice)
    frame2 = Toplevel()
    im=PILImage.open(name)
    img=ImageTk.PhotoImage(im)
    Label(frame2,image=img).pack()
    Button(frame2,text="Quit",command=quit).pack(side=BOTTOM)
    mainloop()
    

def prob_draw(positive,negative,fptr):
    labels=['Predicted to have heart failure','Predicted healthy']
    X=[positive,negative]  
    colors = ['firebrick', 'olive']
    fig = plt.figure(figsize=(8, 4))
    plt.pie(X,labels=labels,autopct='%1.2f%%',colors = colors) 
    plt.title("Predicted results: %s" % fptr)
    fig.savefig("Figure/%s_PieChart.png" % fptr)
    return "Figure/%s_PieChart.png" % fptr


if __name__ == '__main__':
    df = pd.read_csv('heart.csv')
    data = read_data(df)
    EDA(data)
    # cleaned_data = data_clean(data)
    # t_SNE(cleaned_data)
    # logisticRegression(data)
    # RandomForest(data)
    # decision_tree(data)
    # gaussian_nb(data)
    # new_patient_info = input_gui()
    # visual_gui(new_patient_info)