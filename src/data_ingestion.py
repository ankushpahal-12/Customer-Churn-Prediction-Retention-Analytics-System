import os
import pandas as pd
class DataIngestion:
    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path
        
    def ingestion_data(self):
        print("Starting the Data to ingestion process....")
        df=pd.read_csv(self.input_path)
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        df.to_csv(self.output_path, index=False)
        return df
        