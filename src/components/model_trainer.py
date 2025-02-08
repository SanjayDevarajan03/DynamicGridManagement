import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, f1_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_reg_models

@dataclass
class ModelTrainerConfig:
    reg_trainer_model_file_path = os.path.join("artifacts", "reg_model.pkl")
    clf_trainer_model_file_path = os.path.join("artifacts", "clf_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_reg_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test  = (
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            reg_models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            """clf_models = {
                "KNeighborsClassifier": KNeighborsClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "SVC": SVC(),
                "LogisticRegression": LogisticRegression()
            }"""

            reg_params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"]
                },
                "Random Forest": {
                    "n_estimators": [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "KNeighbors Regressor": {  # Fixed parameters for KNeighborsRegressor
                    "n_neighbors": [5, 15, 45, 90],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"],  # Changed from 'algorithms' to 'algorithm'
                    "metric": ["minkowski", "euclidean", "manhattan"]
                },
                "CatBoosting Regressor": {
                    "depth": [6,8,10],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate":[0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8,16,32,64,128,256]
                }
            }


            model_report:dict=evaluate_reg_models(X_train=X_train,y_train=y_train, X_test=X_test,y_test=y_test,models=reg_models, param=reg_params)

            ## To get model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = reg_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.reg_trainer_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_clf_model_trainer(self, train_array, test_array):
        try: 
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = [
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            ]

            clf_models = {

            }

        except Exception as e:
            raise CustomException(e, sys)
