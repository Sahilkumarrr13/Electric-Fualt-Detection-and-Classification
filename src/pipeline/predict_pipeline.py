import sys 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:  
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path) 
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
                 Ia: float,
                 Ib: float,
                 Ic: float,
                 Va: float,
                 Vb: float,
                 Vc: float,
                 ):
        
        self.Ia = Ia
        self.Ib = Ib
        self.Ic = Ic
        self.Va = Va
        self.Vb = Vb
        self.Vc = Vc
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                    "Ia": [self.Ia],
                    "Ib": [self.Ib],
                    "Ic": [self.Ic],
                    "Va": [self.Va],
                    "Vb": [self.Vb],
                    "Vc": [self.Vc],
                }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)