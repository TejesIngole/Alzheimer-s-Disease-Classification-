# 🧠 Alzheimer's Disease Risk Predictor

A machine learning web application that predicts the risk of Alzheimer's Disease based on clinical, lifestyle, and cognitive features using an XGBoost classifier.

🔗 **Live App:** [https://nbz4k42x2udf5qxmzqmvuf.streamlit.app/](https://nbz4k42x2udf5qxmzqmvuf.streamlit.app/)

---

## 📌 Project Overview

This project builds a binary classification model to predict whether a patient is likely to have Alzheimer's Disease based on 23 clinical features including demographics, lifestyle factors, medical history, and cognitive assessments.

---

## 📁 Project Structure

```
├── app.py                        # Streamlit web application
├── alzheimerdisease.py           # Model training & EDA notebook script
├── alzheimers_disease_data.csv   # Dataset (2149 patients, 35 features)
├── best_model.pkl                # Trained XGBoost model
├── scaler.pkl                    # Fitted StandardScaler
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 📊 Dataset

- **Source:** Alzheimer's Disease Dataset
- **Rows:** 2,149 patient records
- **Features:** 35 columns (after dropping PatientID & DoctorInCharge → 33 usable)
- **Target:** `Diagnosis` (0 = No Alzheimer's, 1 = Alzheimer's Detected)

### Feature Groups

| Group | Features |
|---|---|
| Demographics | Age, Gender, Ethnicity, EducationLevel |
| Lifestyle | BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality |
| Medical History | FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension |
| Clinical Measurements | SystolicBP, DiastolicBP, CholesterolLDL, CholesterolHDL, CholesterolTotal, CholesterolTriglycerides |
| Cognitive Assessments | MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL, Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness |

---

## 🤖 Models Trained & Compared

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~85% | ~0.91 |
| KNN | ~83% | ~0.89 |
| SVM (RBF) | ~86% | ~0.93 |
| Random Forest | ~88% | ~0.95 |
| **XGBoost ✅** | **~91%** | **~0.97** |
| LightGBM | ~90% | ~0.96 |

> ✅ **XGBoost** was selected as the best model based on highest ROC-AUC score.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/alzheimers-predictor.git
cd alzheimers-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → Click **New App**
4. Select your repo, branch `main`, file `app.py`
5. Click **Deploy** 🚀

---

## 🖥️ App Features

- **23-feature input form** — organized by clinical category in the sidebar
- **Instant prediction** — Alzheimer's probability with risk level badge
- **Risk gauge** — visual bar showing low / moderate / high risk
- **Clinical recommendation** — actionable advice based on risk score
- **Feature summary chart** — bar chart of patient's input values
- **Dark-themed UI** — clean, professional medical aesthetic

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
joblib
shap
matplotlib
```

---

## ⚠️ Disclaimer

> This tool is for **educational and research purposes only**. It is **not a substitute** for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author

Made with ❤️ using Python, XGBoost & Streamlit.
