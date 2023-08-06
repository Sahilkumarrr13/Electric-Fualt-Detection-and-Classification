import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Spliting training and test input data')
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "SVM Model": SVC(C = 1000),
                "Decision Tree": DecisionTreeClassifier(criterion='gini',ccp_alpha=0.0012),
                "Random Forest": RandomForestClassifier(max_depth=6, min_samples_leaf=10, min_samples_split=15,n_estimators=10),
                "KNN": KNeighborsClassifier(leaf_size= 1, n_neighbors= 5,p= 1)
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException('No Best Model Found')
            
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            score = np.mean(cross_val_score(best_model, X_test, y_test, cv = 3))
            return score   
            
        except Exception as e:
            raise CustomException(e, sys)