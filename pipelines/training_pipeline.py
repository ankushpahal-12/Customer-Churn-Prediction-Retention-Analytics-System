from src.data_analysis import DataAnalysis
from src.data_ingestion import DataIngestion
from src.data_transform import DataTransform
from src.data_visualization import DataVisualization

from model.model_saver import ModelSaver
from model.model_trainer import ModelTrainer
from model.model_selector import ModelSelector


import sys
from utils.logger import Logger
from utils.exception import CustomException

class TrainingPipeline:

    def __init__(self):
        self.raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        self.processed_path = "data/processed/processed_telco_data.csv"

    def run_pipeline(self):
        logger = Logger.setup_logger()
        try:
            logger.info("Starting Training Pipeline...")

            # 🔹 Step 1: Data Ingestion
            ingestion = DataIngestion(self.raw_path, self.processed_path)
            df = ingestion.ingestion_data()
            logger.info("Data Ingestion Completed...")
            
            # 🔹 Step 2: Data Analysis
            analysis = DataAnalysis(df)
            analysis.data_analysis()
            logger.info("Data Analysis Completed...")
            
            # step 3 Data Visulaization
            visualizer = DataVisualization(df)
            visualizer.generate_all_plots()
            logger.info("Data Visualization Completed...")
            
            # 🔹 Step 4: Data Transformation
            transform = DataTransform(df,target_column="Churn")
            X_train, X_test, y_train, y_test, preprocessor ,pca = transform.transform_data()
            logger.info("Data Transformation Completed...")
            
            # step 5: Model Trainer
            trainer=ModelTrainer(X_train,y_train,X_test,y_test)
            results=trainer.train_models()
            logger.info("Model Trainer Completed...")
            
            # step 6: Model Selector
            selector=ModelSelector(results)
            best_model_name,best_model,best_metrics=selector.select_best_model()
            logger.info("Model Selector Completed...")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Best Metrics: {best_metrics}")
            
            # step 7 : Model Saver
            saver=ModelSaver(
                model_path=f"artifacts/models/{best_model_name}_model.pkl",
                preprocessor_path="artifacts/models/preprocessor.pkl",
                metrics_path="artifacts/models/metrics.json",
                
                )
            saver.save_model(best_model)
            saver.save_preprocessor(preprocessor)
            saver.save_metrics(best_metrics)
            saver.save_pca(pca)
            logger.info("Model Saver Completed...")
            
            
            
            logger.info("Training Pipeline Completed...")
        except Exception as e:
            logger.error("Error occurred during training pipeline execution.")
            raise CustomException(e, sys)




if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
