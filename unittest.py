from typing_extensions import ParamSpecArgs
import Heart_Failure_Prediction
import unittest
import pandas as pd

class TestStringMethods(unittest.TestCase):

    def read(self):
        data= Heart_Failure_Prediction.read_data('heart.csv')
        result=data['Age'][0]
        self.assertEqual(result,40)
        
    def EDA_test():
        pass

    def kmeans_find_cluster_test():
        pass

    def para_coor_test():
        pass

    def t_SNE_test():
        pass

    def plot_confusion_matrix_test():
        pass
    
    def ROC_curve_test():
        pass
    
    def Scores_test():
        pass

    def rocauc_score_test():
        pass
    
    def PR_curve_test():
        pass
    
    def logisticRegression_tset():
        pass
    
    def RandomForest_test():
        pass
    
    def decision_tree_test():
        pass
    
    def gaussian_nb_test():
        pass
    
    def plot_learning_curve_test():
        pass

    def input_gui_test(self):
        newp= Heart_Failure_Prediction.input_gui()
        df = pd.read_csv('new_patient.csv')
        self.assertEqual(newp,df)


if __name__ == '__main__':
    unittest.main()