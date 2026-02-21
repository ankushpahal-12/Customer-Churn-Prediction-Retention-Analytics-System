import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataVisualization:

    def __init__(self, df):
        self.df = df
        os.makedirs("artifacts/plots", exist_ok=True)

    def correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        numeric_df = self.df.select_dtypes(include=["number"])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("artifacts/plots/correlation_heatmap.png")
        plt.close()

    def churn_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(x="Churn", data=self.df)
        plt.title("Churn Distribution")
        plt.tight_layout()
        plt.savefig("artifacts/plots/churn_distribution.png")
        plt.close()

    def monthly_charges_vs_churn(self):
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=self.df)
        plt.title("Monthly Charges vs Churn")
        plt.tight_layout()
        plt.savefig("artifacts/plots/monthly_charges_vs_churn.png")
        plt.close()

    def tenure_vs_churn(self):
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Churn", y="tenure", data=self.df)
        plt.title("Tenure vs Churn")
        plt.tight_layout()
        plt.savefig("artifacts/plots/tenure_vs_churn.png")
        plt.close()

    def generate_all_plots(self):
        self.correlation_heatmap()
        self.churn_distribution()
        self.monthly_charges_vs_churn()
        self.tenure_vs_churn()
        print("All analysis plots saved successfully.")
