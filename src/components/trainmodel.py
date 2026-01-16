import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from src.exception import Customexception
from src.logger import logging
from catboost import CatBoostRegressor
from src.util import save_object,evaluate_models

@dataclass
class Model_trainer_config:
    model_trained_path=os.path.join("Artifacts","train_model.pkl")
class ModelTraining:
    def __init__(self):
        self.model_trainer_config=Model_trainer_config()
    def initiateTrain(self,train_arr,test_arr):
        logging.info('splititng into training and test data')
        try:
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
                }
            report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            ## To get best model score from dict
            best_model_score = max(sorted(report.values()))

            ## To get best model name from dict
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info(" not that best model found")

            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.model_trained_path,
                obj=best_model
            )
            predict=best_model.predict(X_test)
            score=r2_score(y_test,predict)
            print(y_test[:5])
            print(predict[:5])
            return score
        except Exception as e:
            raise Customexception(e,sys)
            