
import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs("results", exist_ok=True)

# Load saved objects
preprocessor = joblib.load("results/preprocessor.pkl")
model = joblib.load("results/xgboost_model.pkl")

# Transform data
X_train_processed = pd.DataFrame(
    preprocessor.transform(X_train),
    columns=preprocessor.get_feature_names_out()
)

X_test_processed = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=preprocessor.get_feature_names_out()
)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_processed)

print("SHAP values shape:", shap_values.values.shape)

# Handle binary classification output
if shap_values.values.ndim == 3:
    positive_shap = shap_values.values[:, :, 1]

elif shap_values.values.ndim == 2:
    positive_shap = shap_values.values

else:
    raise ValueError(
        f"Unexpected SHAP dimensions: {shap_values.values.shape}"
    )

# Generate summary plot
plt.figure(figsize=(10, 7))

shap.summary_plot(
    positive_shap,
    X_test_processed,
    show=False
)

plt.savefig(
    "results/xgboost_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Calculate feature importance
importance_df = pd.DataFrame({
    "Feature": X_test_processed.columns,
    "Mean_SHAP": np.abs(positive_shap).mean(axis=0)
}).sort_values("Mean_SHAP", ascending=False)

print("\nXGBoost SHAP Feature Importance:")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    "results/xgboost_shap_importance.csv",
    index=False
)

# Plot importance
plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"].iloc[::-1],
    importance_df["Mean_SHAP"].iloc[::-1]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.title("XGBoost SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    "results/xgboost_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nXGBoost SHAP analysis completed successfully!")
print("Plots and CSV saved in results folder.")
