from dataclasses import dataclass
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    reg_preprocessor_obj_file_path = os.path.join('artifacts', 'reg_preprocessor.pkl')
    clf_preprocessor_obj_file_path = os.path.join('artifacts', 'clf_preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns=["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]

            num_pipeline = Pipeline(
                steps = [
                    ("inputer", SimpleImputer(strategy="mean")),
                    ("scaler", MinMaxScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_reg_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_columns = ["stab"]

            numerical_columns = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]

            input_feature_train_df = train_df.drop(columns=target_columns, axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=target_columns, axis=1)
            target_feature_test_df = test_df[target_columns]

            print("Target Column Names:", target_columns)
            print("DataFrame Columns:", train_df.columns)


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.reg_preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.reg_preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_reg_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df =pd.read_csv(test_path)

                logging.info("Read train and test data completed.")

                logging.info("Obtaining preprocessing object.")

                preprocessing_obj = self.get_data_transformer_object()

                target_columns = ["stabf"]

                numerical_columns = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]

                input_feature_train_df = train_df.drop(columns=target_columns, axis=1)
                target_feature_train_df = train_df[target_columns]

                input_feature_test_df = test_df.drop(columns=target_columns, axis=1)
                target_feature_test_df = test_df[target_columns]

                print("Target Column Names:", target_columns)
                print("DataFrame Columns:", train_df.columns)


                logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                logging.info(f"Saved preprocessing object.")

                save_object(
                    file_path=self.data_transformation_config.clf_preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.clf_preprocessor_obj_file_path
                )
            
            except Exception as e:
                raise CustomException(e, sys)


