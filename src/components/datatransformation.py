import os
import sys
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import Customexception
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.util import save_object


@dataclass
class datatransform_config:
    preprocessor_obj_path=os.path.join("Artifacts","preprocessor.pkl")
class Datatransformation:
    def __init__(self):
        self.transformation_config=datatransform_config()
    def get_transformation(self):
        logging.info("data Transformation has started")
        try:
           catogorical_features= ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
           numeric_features=['reading score','writing score']
           numeric_pipline=Pipeline(
               steps=[
                   ('Imputer',SimpleImputer(strategy="median")),
                   ('standardscalar',StandardScaler())
               ]
           )
           catogorical_pipline=Pipeline(
               steps=[
                   ('Imputer',SimpleImputer(strategy="most_frequent")),
                   ('Encoder',OneHotEncoder(handle_unknown="ignore"))
               ]
           )
           logging.info("Transformation pipeline defintion done!!")
           preprocessor=ColumnTransformer(
               [
                   ("numeric_pipeline",numeric_pipline,numeric_features),
                   ("catogorical_pipeline",catogorical_pipline,catogorical_features)
               ]
           )
           logging.info("Data Transformation is done")
           return preprocessor

        except Exception as e:
            raise Customexception(e,sys)
    def initiate_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading the train and test data is done")
            preprocessor_obj=self.get_transformation()
            target_column="math score"
            input_train_df=train_df.drop(target_column,axis=1)
            input_train_target_df=train_df[target_column]
            input_test_df=test_df.drop(target_column,axis=1)
            input_test_target_df=test_df[target_column]
            input_train_arr=preprocessor_obj.fit_transform(input_train_df)
            input_test_arr=preprocessor_obj.transform(input_test_df)
            train_arr=np.c_[ 
                input_train_arr,np.array(input_train_target_df)
            ]
            test_arr=np.c_[ 
                input_test_arr,np.array(input_test_target_df)
            ]
            save_object(
                file_path=self.transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            logging.info("saving the pickle file is done")
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise Customexception(e,sys)    
