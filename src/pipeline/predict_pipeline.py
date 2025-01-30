import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path=os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)
        

    class CustomData:
        def __init__(self, 
            tau1: float,
            tau2: float,
            tau3: float,
            tau4: float,
            p1: float,
            p2: float,
            p3: float,
            p4: float,
            g1: float,
            g2: float,
            g3: float,
            g4: float):

            #for i in range(1, 5):
             #   self.tau
            self.tau1 = tau1
            self.tau2 = tau2
            self.tau3 = tau3
            self.tau4 = tau4

            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4

            self.g1 = g1
            self.g2 = g2
            self.g3 = g3
            self.g4 = g4

        def get_data_as_data_frame(self):
            try:
                custom_data_input_dict = {
                    "tau1":[self.tau1],
                    "tau2":[self.tau2],
                    "tau3":[self.tau3],
                    "tau4":[self.tau4],
                    "p1":[self.p1],
                    "p2":[self.p2],
                    "p3":[self.p3],
                    "p4":[self.p4],
                    "g1":[self.g1],
                    "g2":[self.g2],
                    "g3":[self.g3],
                    "g4":[self.g4]
                }

                return pd.DataFrame(custom_data_input_dict)
            except Exception as e:
                raise CustomException(e, sys)