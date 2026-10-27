"""
Compute SHAP values for the multimodal Random Forest on Fold 10 test data.
Exports:
  1. shap_feature_importance.csv  - mean |SHAP| per feature (216 rows)
  2. shap_lead_aggregation.csv    - aggregated by ECG lead + clinical group
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import shap

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
sys.path.insert(0, BASE)

# 1. Load model and data
pipeline = joblib.load(os.path.join(MODELS_DIR, "ecg_pipeline.pkl"))
calibrated_model = pipeline.model

# Extract the base RandomForestClassifier from CalibratedClassifierCV
base_rf = calibrated_model.calibrated_classifiers_[0].estimator
print(f"Base model: {type(base_rf).__name__}")
print(f"n_estimators: {base_rf.n_estimators}")

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
print(f"X_test shape: {X_test.shape}")

# 2. Load feature names
with open(os.path.join(MODELS_DIR, "feature_schema.json")) as f:
    schema = json.load(f)
feature_names = schema["fused_feature_names"]
ecg_feature_names = schema["ecg_feature_names"]
clinical_feature_names = schema["clinical_feature_names"]
print(f"Features: {len(feature_names)} (ECG: {len(ecg_feature_names)}, Clinical: {len(clinical_feature_names)})")

# 3. Compute SHAP values using TreeExplainer (exact for tree models)
print("\nComputing SHAP values (TreeExplainer, exact)...")
explainer = shap.TreeExplainer(base_rf)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values can be:
#   - list [class_0_array, class_1_array]  (older SHAP)
#   - 3D ndarray (n_samples, n_features, 2) (newer SHAP)
# We want class 1 (MI) SHAP values
if isinstance(shap_values, list):
    shap_mi = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_mi = shap_values[:, :, 1]
else:
    shap_mi = shap_values
print(f"SHAP values shape (class 1 / MI): {shap_mi.shape}")

# 4. Mean absolute SHAP per feature
mean_abs_shap = np.mean(np.abs(shap_mi), axis=0)

feat_df = pd.DataFrame({
    "feature_name": feature_names,
    "mean_abs_shap": mean_abs_shap,
})
feat_df = feat_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
feat_df.index.name = "rank"
feat_df.index = feat_df.index + 1

feat_csv = os.path.join(BASE, "shap_feature_importance.csv")
feat_df.to_csv(feat_csv)
print(f"\nSaved per-feature SHAP importance ({len(feat_df)} features) to: {feat_csv}")
print("\nTop 20 features by mean |SHAP|:")
print(feat_df.head(20).to_string())

# 5. Aggregate by ECG lead + clinical
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
feats_per_lead = 17

lead_rows = []
for lead_idx, lead_name in enumerate(lead_names):
    start = lead_idx * feats_per_lead
    end = start + feats_per_lead
    lead_shap = mean_abs_shap[start:end]
    lead_rows.append({
        "group": f"Lead {lead_name}",
        "type": "ECG",
        "n_features": feats_per_lead,
        "mean_abs_shap_sum": float(lead_shap.sum()),
        "mean_abs_shap_avg": float(lead_shap.mean()),
        "top_feature": feature_names[start + np.argmax(lead_shap)],
        "top_feature_shap": float(lead_shap.max()),
    })

# Heart rate (single feature at index 204)
hr_shap = mean_abs_shap[204]
lead_rows.append({
    "group": "Heart Rate",
    "type": "ECG",
    "n_features": 1,
    "mean_abs_shap_sum": float(hr_shap),
    "mean_abs_shap_avg": float(hr_shap),
    "top_feature": "heart_rate",
    "top_feature_shap": float(hr_shap),
})

# Clinical features (indices 205-215)
clin_shap = mean_abs_shap[205:216]
lead_rows.append({
    "group": "Clinical Demographics",
    "type": "Clinical",
    "n_features": 11,
    "mean_abs_shap_sum": float(clin_shap.sum()),
    "mean_abs_shap_avg": float(clin_shap.mean()),
    "top_feature": clinical_feature_names[np.argmax(clin_shap)],
    "top_feature_shap": float(clin_shap.max()),
})

lead_df = pd.DataFrame(lead_rows)
lead_df = lead_df.sort_values("mean_abs_shap_sum", ascending=False).reset_index(drop=True)

lead_csv = os.path.join(BASE, "shap_lead_aggregation.csv")
lead_df.to_csv(lead_csv, index=False)
print(f"\nSaved lead-level SHAP aggregation ({len(lead_df)} groups) to: {lead_csv}")
print("\nLead / group aggregation (sorted by total SHAP contribution):")
print(lead_df.to_string(index=False))

# 6. Summary statistics
total_shap = mean_abs_shap.sum()
ecg_total = mean_abs_shap[:205].sum()
clin_total = mean_abs_shap[205:].sum()
print(f"\n=== Summary ===")
print(f"Total mean |SHAP|: {total_shap:.6f}")
print(f"ECG contribution:  {ecg_total:.6f} ({ecg_total/total_shap*100:.1f}%)")
print(f"Clinical contrib:  {clin_total:.6f} ({clin_total/total_shap*100:.1f}%)")
print(f"Test records:      {X_test.shape[0]}")
print(f"SHAP method:       TreeExplainer (exact)")

if __name__ == "__main__":
    pass
