import pandas as pd
import numpy as np
class DataAnalysis:
    def __init__(self,df):
        self.df=df
    
    def validate(self):
        print("Analyzing the data.... ")
        
        null_values=self.df.isnull().sum().sum()
        if null_values>0:
            print(f"Missing values found{null_values} in the dataset.")
        else:
            print("No null values will find")
    
    def shape(self):
        sh=self.df.shape # it returns the number of  rows and  columns
        print("The shape of the data is: ",sh)
    
    def info(self):
        Info=self.df.info() # it gives the summary of the dataset
        print("info: ",Info)
        
    def describe(self):
        desc=self.df.describe() # it gives the statiscal summary of the numberical columns  (Mean, Std, Min , Max, 25% 50% 70%)
        print("Describe: ",desc)# used for the detecting the skewness outliers and daata distribbution
        
    def unique_values(self):
        unique_vals=self.df.nunique() # counts the uniques values in each column
        print("Unique values: ",unique_vals)# checks for the checking cardinaltiy, Identifying Categorical Variables
        
    def value_counts(self):
        value_counts=self.df.value_counts() # Counts frequency of the each value in the column
        print("Value counts: ",value_counts)# used for the checking imbalance, Target distribution
        
    def column_missing(self):
        column_missing_values=self.df.isnull().sum()# it gives missing values per column
        print("Missing Values per column: ",column_missing_values)
        
    def data_analysis(self):
        self.validate()
        self.shape()
        self.info()
        self.describe()
        self.unique_values()
        self.value_counts()
        self.column_missing()


