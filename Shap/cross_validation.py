
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

# -----------------------------
# 1. Load dataset
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

# -----------------------------
# 2. Preprocessing pipeline
# -----------------------------

numeric_transformer = SimpleImputer(
    strategy="median"
)

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# 3. Define models
# -----------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# 4. Cross-validation setup
# -----------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc"
}

# -----------------------------
# 5. Evaluate models
# -----------------------------

results = []

for name, pipeline in [
    ("Random Forest", rf_pipeline),
    ("XGBoost", xgb_pipeline)
]:

    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    row = {"Model": name}

    for metric in scoring:

        values = scores[f"test_{metric}"]

        row[f"{metric}_mean"] = np.mean(values)
        row[f"{metric}_std"] = np.std(values)

    results.append(row)

# -----------------------------
# 6. Display results
# -----------------------------

results_df = pd.DataFrame(results)

print("\n5-FOLD CROSS-VALIDATION RESULTS")
print(results_df.to_string(index=False))

# -----------------------------
# 7. Save results
# -----------------------------

results_df.to_csv(
    "results/cross_validation_results.csv",
    index=False
)

print("\nCross-validation completed successfully!")
