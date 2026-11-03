
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
# 1. Load dataset and saved model
# -----------------------------

df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

categorical_features = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

numeric_features = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs("results", exist_ok=True)

preprocessor = joblib.load("results/preprocessor.pkl")
model = joblib.load("results/random_forest_model.pkl")

# -----------------------------
# 2. Transform test data
# -----------------------------

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

# -----------------------------
# 3. Calculate SHAP values
# -----------------------------

background = X_train_processed.sample(
    n=min(100, len(X_train_processed)),
    random_state=42
)

explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)

shap_values = explainer(X_test_processed)

# Select positive class: HeartDisease = 1
positive_shap = shap_values[:, :, 1]

# -----------------------------
# 4. Create SHAP DataFrame
# -----------------------------

shap_df = pd.DataFrame(
    np.abs(positive_shap.values),
    columns=feature_names
)

# -----------------------------
# 5. Group encoded features
# -----------------------------

grouped_importance = {}

for column in shap_df.columns:

    original_feature = column.split("__", 1)[-1]

    for feature in categorical_features:

        if original_feature.startswith(feature + "_"):
            original_feature = feature
            break

    grouped_importance.setdefault(
        original_feature, []
    ).append(shap_df[column].values)

grouped_results = {
    feature: np.mean(np.sum(values, axis=0))
    for feature, values in grouped_importance.items()
}

importance_df = pd.DataFrame(
    grouped_results.items(),
    columns=["Feature", "Mean_SHAP"]
).sort_values(
    "Mean_SHAP",
    ascending=True
)

print("\nGrouped SHAP Feature Importance:")
print(importance_df.to_string(index=False))

# -----------------------------
# 6. Save results
# -----------------------------

importance_df.to_csv(
    "results/grouped_shap_importance.csv",
    index=False
)

# -----------------------------
# 7. Plot grouped importance
# -----------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Mean_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Clinical Features")
plt.title("Grouped Clinical Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/grouped_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGrouped SHAP analysis completed successfully!")
print("Results saved in the results folder.")
