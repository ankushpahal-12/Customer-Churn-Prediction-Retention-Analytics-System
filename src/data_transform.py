import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
class DataTransform:
    def __init__(self,df,target_column):
        self.df=df
        self.target_column=target_column
        
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
                        
    def handle_outliers(self):
        num_cols=self.df.select_dtypes(include="number").columns
        for col in num_cols:
            Q1=self.df[col].quantile(0.25) # quantile it retruns the percentile values and it is used in the iqr outlier detection 
            Q2=self.df[col].quantile(0.75)
            IQR=Q2-Q1
            lower_bound=Q1-1.5*IQR
            upper_bound=Q2+1.5*IQR
                
            outliers=((self.df[col]<lower_bound)| 
                      (self.df[col]>upper_bound)).sum()
            
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
    def delete_column(self):
        if "customerID" in self.df.columns:
            self.df.drop("customerID",axis=1,inplace=True)
        if "TotalCharges" in self.df.columns:
            self.df["TotalCharges"]=pd.to_numeric(
                self.df["TotalCharges"],errors="coerce"
                )
        
        
        
    def transform_data(self):

        print("Starting To Transform the data....")

        # Step 1: Delete unwanted columns
        self.delete_column()

        # Step 2: Feature Engineering (BEFORE splitting)

        self.df["tenure_group"] = pd.cut(
            self.df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
        )

        self.df["avg_monthly_cost"] = (
            self.df["TotalCharges"] / (self.df["tenure"] + 1)
        )

        self.df["high_monthly_charge"] = (
            self.df["MonthlyCharges"] > self.df["MonthlyCharges"].median()
        ).astype(int)
        
        self.df["long_term_customer"]=(
            self.df["tenure"]>24
        ).astype(int)
        

        self.df["is_month_to_month"] = (
            self.df["Contract"] == "Month-to-month"
        ).astype(int)

        # Step 3: Separate features and target
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column].map({"Yes": 1, "No": 0})

        # Step 4: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Step 5: Identify column types (IMPORTANT: use X_train)
        num_cols = X_train.select_dtypes(include="number").columns
        cat_cols = X_train.select_dtypes(include="object").columns

        # Step 6: Pipelines
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols)
            ]
        )

        # Step 7: Transform
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # n_components=[0.90,0.92,0.95,0.98]
        pca=PCA(n_components=0.95,random_state=42)
        X_train=pca.fit_transform(X_train)
        X_test=pca.transform(X_test)
        X_train=pd.DataFrame(X_train,columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        X_test=pd.DataFrame(X_test,columns=[f"feature_{i}" for i in range(X_test.shape[1])])

        print("Transformation of the data is complete...")

        return X_train, X_test, y_train, y_test, preprocessor,pca
