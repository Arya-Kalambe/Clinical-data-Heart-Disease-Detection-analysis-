
import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load dataset
# -----------------------------

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

# -----------------------------
# 2. Load model and preprocessor
# -----------------------------

preprocessor = joblib.load("results/preprocessor.pkl")
model = joblib.load("results/xgboost_model.pkl")

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    preprocessor.transform(X_train),
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=feature_names
)

# -----------------------------
# 3. Calculate SHAP values
# -----------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_test_processed)

print("SHAP shape:", shap_values.values.shape)

if shap_values.values.ndim == 3:
    positive_shap = shap_values.values[:, :, 1]
elif shap_values.values.ndim == 2:
    positive_shap = shap_values.values
else:
    raise ValueError("Unexpected SHAP output dimensions")

# -----------------------------
# 4. Group encoded features
# -----------------------------

categorical_features = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

grouped_values = {}

for i, column in enumerate(feature_names):

    original_feature = column.split("__", 1)[-1]

    for feature in categorical_features:
        if original_feature.startswith(feature + "_"):
            original_feature = feature
            break

    grouped_values.setdefault(
        original_feature, []
    ).append(i)

# Sum absolute SHAP values per patient
grouped_results = {}

for feature, indices in grouped_values.items():

    patient_values = np.abs(
        positive_shap[:, indices]
    ).sum(axis=1)

    grouped_results[feature] = patient_values.mean()

# -----------------------------
# 5. Create results table
# -----------------------------

importance_df = pd.DataFrame(
    grouped_results.items(),
    columns=["Feature", "Mean_SHAP"]
).sort_values(
    "Mean_SHAP",
    ascending=True
)

print("\nGrouped XGBoost SHAP Importance:")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    "results/xgboost_grouped_shap_importance.csv",
    index=False
)

# -----------------------------
# 6. Plot grouped importance
# -----------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Mean_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Contribution")
plt.title("Grouped XGBoost Clinical Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/xgboost_grouped_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGrouped XGBoost SHAP analysis completed!")
print("CSV and graph saved in results folder.")
