## Background
The project plan to use an [open database](https://www.kaggle.com/fedesoriano/heart-failure-prediction) which contains 918 observations with 12 attributes to establish heart failure prediction models. 
Machine learning algorithms such as logistic regression, random forests are possible choices for model fitting. Also, our program will allow heart failure prediction for new variants of a patient.

## Setup instructions
1) git clone the repo
2) do ```python pip install ``` to install the repo below:  

   **Data Address:**  ```pandas```  ```numpy```  ```joblib``` ```sklearn```  

   **Draw Figures:**  ```Pillow``` ```matplotlib``` ```seaborn```  

   **GUI Interface:** ```tkinter``` 


## How our data look like ?

***The Features***

**Age:** age of the patient [years]

**Sex:** sex of the patient [1: Male, 0: Female]

**ChestPainType:** chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
        
**RestingBP:** resting blood pressure [mm Hg]

**Cholesterol:** serum cholesterol [mm/dl]

**FastingBS:** fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]

**RestingECG:** resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
        
**MaxHR:** maximum heart rate achieved [Numeric value between 60 and 202]

**ExerciseAngina:** exercise-induced angina [1: Yes, 0: No]

**Oldpeak:** oldpeak = ST [Numeric value measured in depression]

**ST_Slope:** the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

**HeartDisease:** output class [1: heart disease, 0: Normal]

***The original dataframe***

<img width="900" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/f56c61355279b259daaa18fd1e766cd7bdaf0d24/Figure/data.png>

## Feature Exploration

***1. The heatmap shows the correlation between features and outcome***

<img width="900" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/feature_exploration.png>

***2. The elbow method result shows the optimal number of clusters***
<img width="900" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/main/Figure/clustering%20compare_k_means.png>

***3.The parallel coordinates results shows potential relationship between features (use 'Age' as an example)***

<img width="1200" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/main/Figure/Parellel_Coordination_Age.png>

***4. The t-SNE results shows the potential relationship between features***

<img width="400" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/main/Figure/t-SNE_analysis.png>

## How our Machine learning model look like ?
***Use logistic regression as an example***

**Precision: 0.893**  

**Recall: 0.876**  

**Accuracy: 0.870**  

**F1 Score: 0.885**  

**ROC-AUC Score: 0.913**  

<img width="500" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/Logistic%20Regression_ROC_curve.png>   

<img width="500" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/Logistic%20Regression_PR_curve.png>      

<img width="500" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/Logistic%20Regression_Learning_curve.png>  

<img width="500" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/Logistic%20Regression_confusion_matrix.png> 


## How can this script help you predict the heart failure?
***Here is the GUI interface***

1) Input a new patient information
<img width="200" height="600" src=https://github.com/lelinz174125/SC_final_project/blob/cc897b481b7660ae6cc51be571370927b215f0f2/Figure/gui_interface1.png>

2) Choose a prediction model 
<img width="500" height="300" src=https://github.com/lelinz174125/SC_final_project/blob/cc897b481b7660ae6cc51be571370927b215f0f2/Figure/gui_interface2.png>

3) Obtain a prediction result
<img width="500" height="325" src=https://github.com/lelinz174125/SC_final_project/blob/cc897b481b7660ae6cc51be571370927b215f0f2/Figure/gui_interface3.png>


## Contributors
This project exists thanks to all the people who contribute. 
 
Lelin Zhong: lzhong6@jhu.edu, lzhong6

Meng Qin: mqin7@jh.edu, mengqinqqq
