## Background
The project plan to use an [open database](https://www.kaggle.com/fedesoriano/heart-failure-prediction) which contains 918 observations with 12 attributes to establish heart failure prediction models. 
Machine learning algorithms such as logistic regression, random forests are possible choices for model fitting. Also, our program will allow heart failure prediction for new variants of a patient.

## Setup instructions
1) git clone the repo
2) do ```python pip install ``` to install the repo below:  

   **Data Address:**  ```pandas```  ```numpy```  ```joblib``` ```sklearn```  

   **Draw Figures:**  ```Pillow``` ```matplotlib``` ```seaborn```  

   **GUI Interface:** ```tkinter``` ``` easygui```   

## How our data look like ?
***The original dataframe***
<img width="900" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/f56c61355279b259daaa18fd1e766cd7bdaf0d24/Figure/data.png>

***The heatmap shows the correlation between features and outcome***
<img width="900" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/feature_exploration.png>


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

<img width="400" height="400" src=https://github.com/lelinz174125/SC_final_project/blob/41ea099e8fb8e47faa847a168ad991b69596d4ce/Figure/Logistic%20Regression_confusion_matrix.png> 


## How can this script help you predict the heart failure?
***Here is the GUI interface***

1) Input a new patient information
<img width="200" height="600" src=https://github.com/lelinz174125/SC_final_project/blob/cc897b481b7660ae6cc51be571370927b215f0f2/Figure/gui_interface1.png>

2) Choose a prediction model 
<img width="500" height="300" src=https://github.com/lelinz174125/SC_final_project/blob/cc897b481b7660ae6cc51be571370927b215f0f2/Figure/gui_interface2.png>

3) Obtain a prediction result
<img width="500" height="350" src=https://github.com/lelinz174125/SC_final_project/blob/cc897b481b7660ae6cc51be571370927b215f0f2/Figure/gui_interface3.png>

## Contributors
This project exists thanks to all the people who contribute.  
Lelin Zhong: lzhong6@jhu.edu, lzhong6
Meng Qin: mqin7@jh.edu, mengqinqqq