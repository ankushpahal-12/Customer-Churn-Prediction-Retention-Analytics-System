import pickle
import os
import numpy as np
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Assuming ModelEvaluation is in a local directory named 'model'
from model.model_evaluation import ModelEvaluation

class ModelTrainer:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_models(self):
        results = {}
        print("Starting model training...")

        # 1. Stratified K-Fold Cross Validation Check (Baseline Recall)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Temporary baseline model for CV check
        baseline_xgb = XGBClassifier(random_state=42, eval_metric="logloss")
        baseline_scores = cross_val_score(
            baseline_xgb, 
            self.x_train, 
            self.y_train, 
            cv=skf, 
            scoring="recall"
        )
        print(f"Initial CV Recall Score: {baseline_scores.mean():.4f}")

        # 2. Tuning Logistic Regression
        print("Tuning Logistic Regression...")
        param_grid = {
            "C": [0.1, 0.5, 1, 2, 3],
            "penalty": ["l1", "l2"]
        }
        base_logistic = LogisticRegression(
            solver="liblinear",
            class_weight={0: 1, 1: 1.2},
            max_iter=2000,
            random_state=42
        )
        grid = GridSearchCV(
            base_logistic,
            param_grid,
            scoring="recall",
            cv=5,
            n_jobs=-1
        )
        grid.fit(self.x_train, self.y_train)
        tuned_logistic = grid.best_estimator_

        # 3. Random Forest Model
        rf_model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        )

        # 4. Tuning XGBoost
        print("Tuning XGBoost...")
        xgb_base = XGBClassifier(
            random_state=42,
            scale_pos_weight=(len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])),
            eval_metric="logloss",
            n_jobs=1
        )
        xgb_param_dist = {
            "n_estimators": [400, 600, 800],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.02, 0.05],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "gamma": [0, 0.2, 0.3],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1, 2, 3]
        }
        xgb_search = RandomizedSearchCV(
            xgb_base,
            xgb_param_dist,
            n_iter=30,
            scoring="recall",
            cv=6,
            n_jobs=4,
            random_state=42,
            verbose=1
        )
        xgb_search.fit(self.x_train, self.y_train)
        xgb_model = xgb_search.best_estimator_

        # 5. Tuning LightGBM
        print("Tuning LightGBM...")
        lgbm_base = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            is_unbalance=True,
            random_state=42,
            n_jobs=1,
            force_col_wise=True,
            verbose=-1
        )
        lgbm_param_dist = {
            "n_estimators": [400, 600, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 70],
            "max_depth": [-1, 5, 10],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "min_child_samples": [10, 20, 30]
        }
        lgbm_search = RandomizedSearchCV(
            lgbm_base,
            lgbm_param_dist,
            n_iter=30,
            scoring="recall",
            cv=6,
            n_jobs=4,
            random_state=42,
        )
        lgbm_search.fit(self.x_train, self.y_train)
        lgbm_model = lgbm_search.best_estimator_

        # 6. Probability Calibration (Using CV=3 to handle Stacking compatibility)
        print("Initializing Calibrated Base Learners...")
        calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
        calibrated_lgbm = CalibratedClassifierCV(lgbm_model, method='isotonic', cv=3)

        # 7. Stacking Ensemble
        print("Training Stacking Ensemble...")
        stacking_model = StackingClassifier(
            estimators=[
                ("lr", tuned_logistic),
                ("rf", rf_model),
                ("xgb", calibrated_xgb),
                ("lgbm", calibrated_lgbm)
            ],
            final_estimator=LogisticRegression(
                C=0.4,
                class_weight={0: 1, 1: 1.2},
                max_iter=2000
            ),
            cv=5,
            passthrough=True
        )

        # 8. Evaluation Loop
        models = {
            "Logistic Regression (Tuned)": tuned_logistic,
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
            "LightGBM": lgbm_model,
            "Stacking Ensemble": stacking_model
        }

        for model_name, model in models.items():
            print(f"Training and Evaluating {model_name}...")
            
            # Fitting the model
            model.fit(self.x_train, self.y_train)

            # Evaluating using your ModelEvaluation class
            evaluator = ModelEvaluation(model, self.x_test, self.y_test)
            metrics = evaluator.evaluate_model()

            results[model_name] = {
                "model": model,
                "metrics": metrics
            }

        print("All models trained successfully.")
        return results

    def save_pca(self, pca):
        os.makedirs("artifacts/models", exist_ok=True)
        with open("artifacts/models/pca.pkl", "wb") as f:
            pickle.dump(pca, f)