class ModelSelector:
    def __init__(self,results,selection_metric="roc_auc_score"):
        """
       results: dictionary returned by model trainer
       selection_metric: metric to select the best model
        """
        self.results=results
        self.selection_metric=selection_metric
    
    def select_best_model(self):
        print(f"Selecting the best model based on {self.selection_metric}.....")
        best_model_name=None
        best_model=None
        best_score=float("-inf")
        best_metrics=None
        
        for model_name,data in self.results.items():
            metrics=data.get("metrics",{})
            # metrics=data["metrics"]
            score=metrics.get(self.selection_metric)
            # score=metrics[self.selection_metric]
            
            if score is not None  and score >best_score:
                best_score=score
                best_model_name=model_name
                best_model=data.get("model")
                # best_model=data["model"]
                best_metrics=metrics
        
        if best_model is None:
            raise ValueError(
                f"No valid model found for {self.selection_metric}"
            )
        
        print(f"Best model: {best_model_name} ")
        print(f"{self.selection_metric}: {best_score}")
        
        return best_model_name, best_model, best_metrics