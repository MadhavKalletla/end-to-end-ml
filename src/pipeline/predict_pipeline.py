import sys
import pandas as pd
from src.util import load_object 
from src.exception import Customexception

class Predictionpipeline:
    def __init__(self):
        pass
    def prediction(self,features):
        try:
            model_path="Atrifacts/train_model.pkl"
            obj_path="Atrifacts/preprocessor.pkl"
            model=load_object(file_path=model_path)
            obj=load_object(file_path=obj_path)
            data_scaled=obj.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Customexception(e,sys)
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        math_score: int,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "math score": [self.math_score],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise Customexception(e,sys)    
        
    