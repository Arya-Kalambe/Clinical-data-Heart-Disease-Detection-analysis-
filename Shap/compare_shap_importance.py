
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load grouped SHAP results
rf = pd.read_csv(
    "results/grouped_shap_importance.csv"
)

xgb = pd.read_csv(
    "results/xgboost_grouped_shap_importance.csv"
)

# Normalize within each model for ranking comparison
rf["RF_Normalized"] = (
    rf["Mean_SHAP"] / rf["Mean_SHAP"].sum()
)

xgb["XGB_Normalized"] = (
    xgb["Mean_SHAP"] / xgb["Mean_SHAP"].sum()
)

# Merge both datasets
comparison = pd.merge(
    rf[["Feature", "RF_Normalized"]],
    xgb[["Feature", "XGB_Normalized"]],
    on="Feature",
    how="outer"
).fillna(0)

comparison = comparison.sort_values(
    "RF_Normalized",
    ascending=True
)

print("\nNORMALIZED SHAP COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    "results/shap_model_comparison.csv",
    index=False
)

# Plot
ax = comparison.set_index("Feature")[
    ["RF_Normalized", "XGB_Normalized"]
].plot(
    kind="barh",
    figsize=(11, 7),
    width=0.75
)

ax.set_xlabel("Normalized Mean Absolute SHAP Importance")
ax.set_title("Random Forest vs XGBoost - Feature Importance")
ax.legend(["Random Forest", "XGBoost"])

plt.tight_layout()

plt.savefig(
    "results/shap_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison graph saved successfully!")
