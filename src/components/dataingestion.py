from sklearn.model_selection import train_test_split
import os 
import sys
from src.exception import Customexception
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.datatransformation import Datatransformation
from src.components.datatransformation import datatransform_config
from src.components.trainmodel import ModelTraining
from src.components.trainmodel import Model_trainer_config

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("Artifacts","train.csv")
    test_data_path=os.path.join("Artifacts","test.csv")
    raw_data_path=os.path.join("Artifacts","raw.csv")
class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_ingestion(self):
        logging.info("Entered into data ingestion method")
        try:
            df=pd.read_csv('notebook\StudentsPerformance.csv')
            logging.info("Reading the dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split has been initiated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise Customexception(e,sys)
if __name__=="__main__":
    obj=Dataingestion()
    train_path,test_path=obj.initiate_ingestion()
    data_trans=Datatransformation()
    train_arr,test_arr,preprocessor_path=data_trans.initiate_transformation(train_path=train_path,test_path=test_path)
    train_obj=ModelTraining()
    print(train_obj.initiateTrain(train_arr=train_arr,test_arr=test_arr))


          
