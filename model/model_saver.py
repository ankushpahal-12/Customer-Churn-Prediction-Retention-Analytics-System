import pickle 
import os
import json

class ModelSaver:
    def __init__(self,model_path,preprocessor_path,metrics_path=None):
        self.model_path=model_path
        self.metrics_path=metrics_path
        self.preprocessor_path=preprocessor_path
        
    def save_model(self,model):
        os.makedirs(os.path.dirname(self.model_path),exist_ok=True)
        with open(self.model_path,"wb") as file:
            pickle.dump(model,file)
        print(f"Model saved successfully{self.model_path}")
        
        
    def save_preprocessor(self,preprocessor):
        os.makedirs(os.path.dirname(self.preprocessor_path),exist_ok=True)
        with open(self.preprocessor_path,"wb") as file:
            pickle.dump(preprocessor,file)
        print(f"Preprocessor saved successfully{self.preprocessor_path}")
    
    def save_metrics(self,metrics):
        if self.metrics_path is not None:
            os.makedirs(os.path.dirname(self.metrics_path),exist_ok=True)
            with open(self.metrics_path,"w") as file:
                json.dump(metrics,file,indent=4)
            print(f"Metrics saved successfully{self.metrics_path}")
            
    def save_pca(self, pca):
        pca_path = os.path.join(
            os.path.dirname(self.model_path),
            "pca.pkl"
        )
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)
        print(f"PCA saved successfully at {pca_path}")
            
        
        