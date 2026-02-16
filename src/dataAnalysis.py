import pandas as pd
import numpy as np
class DataAnalysis:
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=pd.read_csv(self.file_path)
    
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

class MissingDataHandler:
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=pd.read_csv(self.file_path)
        
    def handle_missing_data(self):
        total_size=self.df.size # total number of the rows and the size 
        missing_values=self.df.isnull().sum().sum() # it gives the total missing values in the entire dataset
        missing_percentage=(missing_values/total_size)*100
        
        print(f"Total size of the dataset:{total_size} and the missing percentage is:{missing_percentage:.2f}%")
            
        if missing_percentage>30:
            print("The dataset has more than 30% missing vales we have to drop the dataset.")
            self.df=self.df.dropna() # it ued for the dropping the missing values 
            return self.df
        else:
            print("The dataset has less than 30% missing values we can handle the missing values.")
            num_cols=self.df.select_dtypes(include="number").columns# dtypes tells about the data type of each column in a dataframe select_dtypes is used for which type of data u have to select
            for col in num_cols:
                if self.df[col].isnull().sum()>0:
                    median_value=self.df[col].median() # middle of the values when the data is sorted and used when we  have the outliers
                    """
                    The middle values when data is sorted and used when data is numerical , data has outliers and data is skewed
                    """
                    self.df[col].fillna(median_value,inplace=True)
                    print(f"Missing values in column {col} have been filled with median value: {median_value}")
                
            cat_cols=self.df.select_dtypes(include="object").columns # this is used for selecting the datatypes related to the object
            """
            Mode is used when we have the data type is categorical , text columns and having the labels
            """
            for col in cat_cols:
                if self.df[col].isnull().sum()>0:
                    mode_value=self.df[col].mode()[0]
                    self.df[col].fillna(mode_value,inplace =True)
                    print(f"Missing values in column {col} have been filled with mode value: {mode_value}")
            print("Missing values have been handled successfully.")
            return self.df
        
    def handle_date(self):
        date_cols=self.df.select_dtypes(include=["object"]).colummns
        for col in date_cols:
            try:
                self.df[col]=pd.to_datetime(self.df[col])
                print(f"Column {col} has been converted to datetime format.")
            except Exception as e:
                print(f"Column {col} could not be converted to datetime format. Error: {e}")
                        
    def handle_outliers(self):
        num_cols=self.df.select_dtypes(include="number").colummns
        for col in num_cols:
            Q1=self.df[col].quantile(0.25) # quantile it retruns the percentile values and it is used in the iqr outlier detection 
            Q2=self.df[col].quantile(0.75)
            IQR=Q2-Q1
            lower_bound=Q1-1.5*IQR
            upper_bound=Q2+1.5*IQR
                
            outliers=((self.data[self.df[col]<lower_bound] | self.df[col]>upper_bound)).sum()
            print(f"Column {col} has {outliers} outliers.")
                
            print("we are using the capping meathod")
            self.df[col]=np.where(
                self.df[col]<lower_bound,
                lower_bound,
                np.where(self.df[col]>upper_bound,
                         upper_bound,
                         self.df[col]
                         )
                )
        #self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        print("Outliers handled completed")
        return self.df              
    """
    How do you cap outliers in pandas ?
    We calculate IQR-based lower and upper bounds
    and use pandas .clip() function to replace extreme 
    values while preserving dataset size
    
    Capping is a technique used to handle outliers by replaacing extreme values beyond a certain statiscal boundary like iqr instread of removing them preserving dataset size while reducing the skewness
    """  