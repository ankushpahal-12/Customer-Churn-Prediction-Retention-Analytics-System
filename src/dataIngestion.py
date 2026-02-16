import os
import pandas as pd
class DataIngestion:
    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path
        
    def ingestion_data(self):
        df=pd.read_csv(self.input_path)
        os.makedirs(self.output_path,exist_ok=True)
        raw_path=os.path.join(self.output_path,)
        df.to_csv(raw_path,index=False)
        
        return raw_path
        