# Multimodal Heart Disease Prediction System

AI-based clinical decision-support system that combines **12-lead ECG waveforms at 500 Hz** with **Patient Clinical Demographics** (Age, Sex, Height, Weight, derived BMI, BSA) using **Multimodal Feature Fusion** to predict **Myocardial Infarction (MI)**.

> **Disclaimer:** This is an AI-based research decision-support prototype. It is NOT a certified medical device and must NOT be used as a standalone diagnostic tool.

---

## 🌟 Key System Features

1. **500 Hz High-Resolution Waveform Branch**:
   - 5,000 samples per lead ($10\text{ seconds} \times 500\text{ Hz}$) across 12 leads for all 21,799 PTB-XL recordings (`records500/`).
   - High-pass (0.5 Hz) and low-pass (40 Hz) Butterworth filtering + per-lead Z-score normalization.
   - 205 extracted signal, statistical, rhythm, and frequency features across 12 leads.

2. **Leakage-Safe Clinical Demographics Branch**:
   - Patient metadata: Age, Sex, Height, Weight, derived BMI, BSA, elderly/obesity indicators, and arrhythmia history flags.
   - Missing height/weight values imputed using **Training set (Folds 1–8) medians strictly** (`models/clinical_imputer.json`) to eliminate data leakage.

3. **Multimodal Feature Fusion**:
   - Concatenated feature vector combining ECG domain features (205) and clinical demographic features (11).

4. **Explainable AI (SHAP & 12-Lead Diagnostic Highlighting)**:
   - Sample-level SHAP feature importance extracting top positive (MI Risk) and negative (Normal) diagnostic drivers.
   - Lead-by-lead diagnostic percentage weight calculation, highlighting primary diagnostic lead drivers (e.g. Lead V2–V4 for Anterior MI) directly on the UI dashboard and API payload.

5. **Probability Calibration & Dual Clinical Operating Modes**:
   - Model probabilities calibrated via `CalibratedClassifierCV` (Sigmoid scaling).
   - **🎯 Screening Mode (Threshold 0.20 / 0.35)**: High Sensitivity (**70.36% – 73.06%**) for emergency triage.
   - **🔒 Conservative Mode (Threshold 0.50)**: High Specificity (**93.68%**) and Precision (**67.55%**).
   - Validation-derived uncertainty margin ($\pm 0.08$ around 0.50), flagging predictions with probabilities between 42% and 58% as `"Uncertain — Clinical Review Recommended"`.

---

## 📁 Project Structure

```
ISP-main/
├── Dataset/                    <- PTB-XL dataset (ptbxl_database.csv, records500/, etc.)
├── data/                       <- Generated: features.csv, splits (.npy), metadata CSVs
│   ├── clean_metadata.csv      <- Processed metadata with clean clinical features & 500 Hz file paths
│   ├── features.csv            <- Fused feature matrix
│   └── X_train.npy, X_val.npy, X_test.npy
├── models/                     <- Saved models and serialized pipeline
│   ├── ecg_pipeline.pkl        <- Complete inference pipeline (MultimodalECGPipeline with XAI)
│   ├── best_model.pkl          <- Best trained calibrated model (XGBoost)
│   ├── clinical_imputer.json   <- Saved training set median imputation parameters
│   ├── clinical_threshold.json <- Validation-derived clinical operating threshold (0.20 / 0.35)
│   ├── feature_schema.json     <- Pipeline feature schema metadata
│   └── preprocessing_objects/
│       └── scaler.pkl          <- StandardScaler fitted on training set
├── results/                    <- baseline_experiments.csv, results_summary.json, ROC/PR plots
├── src/
│   ├── inference/
│   │   └── pipeline.py         <- MultimodalECGPipeline class definition (with SHAP XAI)
│   ├── phase1_load_and_label.py
│   ├── phase2_data_cleaning.py
│   ├── phase3_4_preprocess_features.py
│   ├── phase5_split.py
│   ├── phase6_train_models.py
│   ├── phase7_8_pipeline.py
│   ├── phase11_final_evaluation.py
│   ├── phase12_threshold_tuning.py
│   ├── phase13_fn_and_subtype_analysis.py
│   └── run_pipeline.py         <- Master runner script
├── backend/
│   └── main.py                 <- FastAPI backend API with XAI endpoints
└── frontend/
    └── index.html              <- Research UI Dashboard with SHAP cards & lead badges
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
pip install wfdb scipy scikit-learn xgboost catboost imbalanced-learn fastapi uvicorn python-multipart
```

### Step 2 — Run Full End-to-End Pipeline (Master Script)
```bash
python src/run_pipeline.py
```

### Step 3 — Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Step 4 — Open the Frontend Dashboard
Simply open `frontend/index.html` in your web browser.

---

## 📊 Baseline Experiments Results (Untouched Test Set — Fold 10)

| Experiment / Input | Model | Test Accuracy | Test Recall (Sensitivity) | Test Specificity | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clinical Only** | Decision Tree | 71.3% | 34.5% | 79.9% | 0.6805 | 0.2985 |
| **Clinical Only** | Random Forest | 70.9% | 34.5% | 79.4% | 0.6818 | 0.2859 |
| **Clinical Only** | XGBoost | 72.9% | 33.3% | 82.2% | 0.6874 | 0.2954 |
| **Clinical Only** | CatBoost | 70.4% | 39.5% | 77.6% | 0.6946 | 0.2981 |
| **ECG Only** | Decision Tree | 80.1% | 63.6% | 83.9% | 0.8140 | 0.4876 |
| **ECG Only** | Random Forest | 83.5% | 61.2% | 88.7% | 0.8596 | 0.6009 |
| **ECG Only** | XGBoost | 85.6% | 51.8% | 93.5% | 0.8661 | 0.6276 |
| **ECG Only** | CatBoost | 85.2% | 54.2% | 92.4% | 0.8576 | 0.6062 |
| **ECG + Clinical (Fused)** | Decision Tree | 80.4% | 66.8% | 83.6% | 0.8241 | 0.5124 |
| **ECG + Clinical (Fused)** | Random Forest | 84.4% | 61.0% | 89.8% | 0.8643 | 0.6171 |
| **ECG + Clinical (Fused)** ⭐ | **XGBoost** | **86.4%** | **55.2%** | **93.7%** | **0.8728** | **0.6468** |
| **ECG + Clinical (Fused)** | CatBoost | 85.5% | 53.7% | 92.9% | 0.8623 | 0.6253 |

### 🎯 Clinical Operating Modes Comparison (Fold 10 Test Set)

| Operating Mode | Threshold | Test Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **🎯 Screening Mode (High Sensitivity)** | **0.35 / 0.20** | **83.7%** | **70.4% – 73.1%** | **86.8%** | **55.1%** | **0.6176** |
| **🔒 Conservative Mode (Standard)** | **0.50** | **86.4%** | **55.2%** | **93.7%** | **67.6%** | **0.6074** |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check & pipeline status |
| `GET` | `/api/model-info` | Feature schema metadata & benchmark summary |
| `POST` | `/api/predict/array` | Multimodal prediction with SHAP XAI from 500 Hz signal matrix + patient demographics |
| `POST` | `/api/predict/file` | Multimodal prediction with SHAP XAI from uploaded WFDB ZIP archive + patient demographics |

### Example Request (`POST /api/predict/array`)
```json
{
  "signal": [[v1, v2, ..., v12], ...],
  "age": 62,
  "sex": 0,
  "height": 175,
  "weight": 80,
  "threshold": 0.20
}
```

### Example Response (with SHAP Explainability)
```json
{
  "prediction": "MI",
  "mi_probability": 0.7842,
  "non_mi_probability": 0.2158,
  "confidence": 78.4,
  "operating_threshold": 0.20,
  "status": "MI Pattern Detected (Operating Threshold 0.20)",
  "disclaimer": "This is an AI-based research decision-support result and is not a medical diagnosis.",
  "explanation": {
    "highlighted_leads": ["V2", "V3", "V4"],
    "lead_contributions": {
      "Lead V2": { "percentage": 18.2, "is_primary_driver": true },
      "Lead V3": { "percentage": 16.5, "is_primary_driver": true }
    },
    "top_features": [
      {
        "feature": "lead_V3_freq_entropy",
        "label": "Lead V3 Freq Entropy",
        "value": 4.12,
        "contribution": 0.245,
        "direction": "Positive (MI Risk)"
      }
    ]
  }
}
```

---

## 🔒 Patient-Level Splitting & Data Leakage Prevention

- **Patient Separation**: Grouped by `patient_id` using PTB-XL `strat_fold`:
  - **Folds 1–8**: Training set (17,418 records / 28,222 after SMOTE)
  - **Fold 9**: Validation set (2,183 records) — hyperparameter tuning, threshold selection & uncertainty calibration
  - **Fold 10**: Final Test set (2,198 records) — evaluated ONCE at the end
- **Zero Patient Overlap**: $Train \cap Validation \cap Test = \emptyset$.
- **Training-Only Imputation**: Median height/weight values are computed on Folds 1–8 only and stored in `models/clinical_imputer.json`.
