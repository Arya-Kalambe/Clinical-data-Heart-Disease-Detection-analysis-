
import pandas as pd

# Load saved metrics
rf = pd.read_csv("results/model_metrics.csv")
xgb = pd.read_csv("results/xgboost_metrics.csv")

# Convert Random Forest Metric/Score format
rf_scores = dict(zip(rf["Metric"], rf["Score"]))

comparison = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Accuracy": rf_scores["Accuracy"],
        "ROC_AUC": rf_scores["ROC-AUC"]
    },
    {
        "Model": "XGBoost",
        "Accuracy": xgb.loc[0, "Accuracy"],
        "ROC_AUC": xgb.loc[0, "ROC_AUC"]
    }
])

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
