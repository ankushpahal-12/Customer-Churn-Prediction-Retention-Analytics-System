import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
) 

class ModelEvaluation:
    def __init__(self,model,x_test,y_test):
        self.model=model
        self.x_test=x_test
        self.y_test=y_test
    def find_best_threshold(self, y_prob):

        precisions, recalls, thresholds = precision_recall_curve(
            self.y_test, y_prob
        )

        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-8
        )

        best_index = np.argmax(f1_scores)

        best_threshold = thresholds[best_index]
        best_f1 = f1_scores[best_index]

        return best_threshold, best_f1
    def evaluate_model(self):
        y_prob=None
        if hasattr(self.model,"predict_proba"):
            y_prob=self.model.predict_proba(self.x_test)[:,1]
            threshold,best_f1=self.find_best_threshold(y_prob)
            y_pred=(y_prob>=0.46).astype(int)
            roc_auc=roc_auc_score(self.y_test,y_prob)
        else:
            y_pred=self.model.predict(self.x_test)
            roc_auc=None

        # calculate metrices
        metrices={
            "accuracy":float(accuracy_score(self.y_test,y_pred)),
            "precision":float(precision_score(self.y_test,y_pred)),
            "recall":float(recall_score(self.y_test,y_pred)),
            "f1_score":float(f1_score(self.y_test,y_pred)),
            "roc_auc_score":float(roc_auc)if roc_auc is not None else None,
            "confusion_matrix":confusion_matrix(self.y_test,y_pred).tolist(),
            "classification_report":classification_report(self.y_test,y_pred)
        }
        self.plot_confusion_matrix(y_pred)
        if y_prob is not None:
            self.plot_roc_curve(y_prob)
        return metrices
    
    def save_plot(self,filename):
        os.makedirs("artifacts/plots",exist_ok=True)
        path=os.path.join("artifacts/plots",filename)
        plt.savefig(path,bbox_inches="tight")
        plt.close()
        
    def plot_confusion_matrix(self, y_pred):

        conf_matrix = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        self.save_plot("confusion_matrix.png")

    def plot_roc_curve(self, y_prob):

        fpr, tpr, _ = roc_curve(self.y_test, y_prob)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        self.save_plot("roc_curve.png")
        
    def plot_feature_importance(self,model,feature_names):
        importances=model.feature_importances_
        indices=np.argsort(importances)[-15:]
        plt.figure(figsize=(10,6))
        plt.barh(range(len(indices)),importances[indices])
        plt.yticks(range(len(indices)),[feature_names[i] for i in indices])
        plt.title("Top 15 feature Importances")
        plt.tight_layout()
        
        self.save_plot("feature_importance.png")