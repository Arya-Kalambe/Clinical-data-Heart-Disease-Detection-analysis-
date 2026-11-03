
# ============================================================
# EXPLAINABLE HEART DISEASE DETECTION USING RANDOM FOREST + SHAP
# ============================================================

# 1. IMPORT LIBRARIES
import matplotlib
matplotlib.use("Agg")  # Prevent interactive plot windows

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "heart.csv"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("\nStarting Heart Disease SHAP Analysis...")


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ============================================================
# 6. TRANSFORM DATA
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)

# Convert sparse matrices into dense arrays
if hasattr(X_train_transformed, "toarray"):
    X_train_dense = X_train_transformed.toarray()
else:
    X_train_dense = np.asarray(X_train_transformed)

if hasattr(X_test_transformed, "toarray"):
    X_test_dense = X_test_transformed.toarray()
else:
    X_test_dense = np.asarray(X_test_transformed)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

print("\nTotal Transformed Features:", len(feature_names))


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest Model...")

model.fit(
    X_train_dense,
    y_train
)

print("Model Training Completed!")


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test_dense)

y_prob = model.predict_proba(X_test_dense)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n================================")
print("MODEL EVALUATION RESULTS")
print("================================")

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "ROC-AUC"],
    "Score": [accuracy, roc_auc]
})

metrics.to_csv(
    RESULTS_DIR / "model_metrics.csv",
    index=False
)


# ============================================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Heart Disease", "Heart Disease"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("Heart Disease Prediction - Confusion Matrix")

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 10. SHAP EXPLAINER
# ============================================================

print("\nInitializing SHAP Explainer...")

background = shap.sample(
    X_train_dense,
    min(100, len(X_train_dense)),
    random_state=42
)

explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)

print("SHAP Explainer Initialized!")


# ============================================================
# 11. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP Values...")

shap_values = explainer(X_test_dense)

# Select positive class: HeartDisease = 1
if shap_values.values.ndim == 3:

    shap_values_positive = shap_values.values[:, :, 1]

    base_values_positive = shap_values.base_values[:, 1]

else:

    shap_values_positive = shap_values.values

    base_values_positive = shap_values.base_values

# Create SHAP Explanation object with actual feature names
positive_shap = shap.Explanation(
    values=shap_values_positive,
    base_values=base_values_positive,
    data=X_test_dense,
    feature_names=feature_names
)

print("SHAP Values Calculated Successfully!")


# ============================================================
# 12. SHAP FEATURE IMPORTANCE
# ============================================================

print("\nGenerating SHAP Feature Importance...")

shap.plots.bar(
    positive_shap,
    max_display=20,
    show=False
)

fig = plt.gcf()

fig.suptitle("SHAP Feature Importance", fontsize=15)

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. SHAP SUMMARY / BEESWARM PLOT
# ============================================================

print("Generating SHAP Summary Plot...")

shap.plots.beeswarm(
    positive_shap,
    max_display=20,
    show=False
)

fig = plt.gcf()

fig.suptitle("SHAP Summary Plot", fontsize=15)

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. SHAP DEPENDENCE PLOT - AGE
# ============================================================

print("Generating SHAP Dependence Plot...")

shap.dependence_plot(
    "num__Age",
    positive_shap.values,
    X_test_dense,
    feature_names=feature_names,
    show=False
)

fig = plt.gcf()

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "shap_age_dependence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. INDIVIDUAL PATIENT WATERFALL PLOT
# ============================================================

print("Generating Patient Waterfall Plot...")

patient_index = 0

patient_explanation = positive_shap[patient_index]

shap.plots.waterfall(
    patient_explanation,
    max_display=15,
    show=False
)

fig = plt.gcf()

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "shap_patient_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Individual Patient Explanation Generated!")


# ============================================================
# 16. GROUPED SHAP FEATURE IMPORTANCE
# ============================================================

print("\nCalculating Grouped SHAP Importance...")

# Calculate absolute SHAP values
shap_df = pd.DataFrame(
    np.abs(positive_shap.values),
    columns=feature_names
)

grouped_importance = {}

for column in shap_df.columns:

    original_feature = column.split("__", 1)[-1]

    # Combine one-hot encoded categorical variables
    for feature in categorical_features:

        if original_feature.startswith(feature + "_"):

            original_feature = feature

            break

    grouped_importance.setdefault(
        original_feature,
        []
    ).append(shap_df[column].values)


# Sum encoded feature contributions for each patient,
# then calculate the average across patients.
grouped_results = {

    feature: np.mean(
        np.sum(values, axis=0)
    )

    for feature, values in grouped_importance.items()

}


# Convert into DataFrame
importance_df = pd.DataFrame(
    grouped_results.items(),
    columns=["Feature", "Mean_SHAP"]
).sort_values(
    "Mean_SHAP",
    ascending=False
)

print("\nGrouped SHAP Feature Importance:")
print(importance_df.to_string(index=False))


# Save grouped results
importance_df.to_csv(
    RESULTS_DIR / "grouped_shap_importance.csv",
    index=False
)


# ============================================================
# 17. GROUPED SHAP VISUALIZATION
# ============================================================

print("\nGenerating Grouped SHAP Plot...")

plot_df = importance_df.sort_values(
    "Mean_SHAP",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    plot_df["Feature"],
    plot_df["Mean_SHAP"]
)

ax.set_xlabel("Mean Absolute SHAP Value")

ax.set_ylabel("Clinical Features")

ax.set_title("Grouped Clinical Feature Importance")

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "grouped_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 18. SAVE TRAINED MODEL AND PREPROCESSOR
# ============================================================

joblib.dump(
    model,
    RESULTS_DIR / "random_forest_model.pkl"
)

joblib.dump(
    preprocessor,
    RESULTS_DIR / "preprocessor.pkl"
)


# ============================================================
# 19. FINAL STATUS
# ============================================================

print("\n======================================")
print("HEART DISEASE SHAP ANALYSIS COMPLETED")
print("======================================")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("ROC-AUC:", round(roc_auc, 4))

print("\nAll graphs and results saved in:")

print(RESULTS_DIR)

print("\nGenerated Files:")

for file in sorted(RESULTS_DIR.iterdir()):

    print("-", file.name)

print("\nSUCCESS! All analysis completed.")
