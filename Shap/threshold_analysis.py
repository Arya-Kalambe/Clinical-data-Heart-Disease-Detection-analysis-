
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

categorical_features = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope"
]

numeric_features = [
    "Age", "RestingBP", "Cholesterol",
    "FastingBS", "MaxHR", "Oldpeak"
]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Out-of-fold probabilities: each sample is predicted by a model
# that did not train on that sample.
probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

results = []

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0,
        "F1_Score": f1_score(y, predictions, zero_division=0),
        "False_Negatives": fn,
        "False_Positives": fp
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD ANALYSIS")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "results/threshold_analysis.csv",
    index=False
)

print("\nThreshold analysis completed!")
