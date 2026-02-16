from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder, LabelEncoder
import pandas as pd
class DataTransform: # data Transform means  converting raw cleaned data into a format suitable for machine learning models
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=pd.read_csv(self.file_path)
        
    def encode_categorical_data(self):
        self.ohe=OneHotEncoder()
        self.lb=LabelEncoder()
        
        cat_cols=self.df.select_dtypes(include="object")
        # for cat_cols in self.df
        
        