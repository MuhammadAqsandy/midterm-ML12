[README_ML.md](https://github.com/user-attachments/files/27901529/README_ML.md)
# 🤖 Midterm Machine Learning — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![Optuna](https://img.shields.io/badge/Optuna-3.4+-purple)
![MLflow](https://img.shields.io/badge/MLflow-2.9+-red?logo=mlflow)

---

## 👤 Identitas

| | |
|---|---|
| **Nama** | Muhammad Aqsandy J Iskandar |
| **Kelas** | TK-46-02 |
| **NIM** | 1103220214 |
| **Mata Kuliah** | Machine Learning |

---

## 📌 Tujuan Repository

Repository ini berisi implementasi end-to-end pipeline Machine Learning untuk dua task utama:

1. **Task 1 — Fraud Detection (Klasifikasi)** : Memprediksi probabilitas transaksi online yang bersifat fraudulent menggunakan dataset IEEE-CIS Fraud Detection.
2. **Task 2 — Song Release Year Prediction (Regresi)** : Memprediksi tahun rilis lagu berdasarkan fitur audio menggunakan dataset MSD (Million Song Dataset).

---

## 🗂️ Struktur Repository

```
midterm-machine-learning/
│
├── 📁 task1-fraud-detection/
│   ├── fraud_detection_colab.ipynb   ← Notebook utama (Klasifikasi)
│   └── README_task1.md
│
├── 📁 task2-regression/
│   ├── regresi_colab.ipynb           ← Notebook utama (Regresi)
│   └── README_task2.md
│
└── README.md                         ← File ini
```

---

## 🔍 Task 1 — Fraud Detection (Klasifikasi)

### Dataset
- **Sumber** : [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- **File** : `train_transaction.csv` (+ opsional `train_identity.csv`)
- **Target** : `isFraud` (0 = Legit, 1 = Fraud)

### Pipeline
```
Load Data → EDA → Preprocessing → Optuna Tuning → Train → Evaluate → MLflow
```

### Models
| Model | Deskripsi |
|---|---|
| **XGBoost Classifier** | Gradient boosting berbasis pohon, sangat efektif untuk data tabular |
| **LightGBM Classifier** | Gradient boosting berbasis leaf-wise, lebih cepat untuk dataset besar |

### Preprocessing
- Drop kolom dengan missing value > 90%
- Label Encoding untuk fitur kategorikal
- Median Imputation untuk missing values
- StandardScaler untuk normalisasi fitur
- **SMOTE** untuk menangani class imbalance (fraud hanya ~3.5%)

### Hyperparameter Tuning
- **Optuna** dengan 30 trials per model
- Optimasi metrik: **AUC-ROC** via 5-Fold Stratified Cross-Validation

### Metrics Evaluasi
| Metric | Deskripsi |
|---|---|
| **AUC-ROC** | Area Under ROC Curve — metrik utama |
| **F1-Score** | Harmonic mean Precision & Recall |
| **Precision** | Dari semua prediksi fraud, berapa yang benar |
| **Recall** | Dari semua fraud aktual, berapa yang terdeteksi |

### Tracking
- Semua eksperimen dilacak dengan **MLflow** (params, metrics, model artifacts)

---

## 📈 Task 2 — Song Release Year Prediction (Regresi)

### Dataset
- **Sumber** : Midterm Regression Dataset
- **File** : `midterm-regresi-dataset.csv`
- **Target** : Kolom pertama = tahun rilis lagu (continuous value)
- **Fitur** : `feature_1` s/d `feature_N` — karakteristik audio lagu (timbre, dll)

### Pipeline
```
Load Data → EDA → Preprocessing → Optuna Tuning → Train → Evaluate → LIME → MLflow
```

### Models
| Model | Deskripsi |
|---|---|
| **Ridge Regression** | Model linear dengan regularisasi L2 — baseline |
| **XGBoost Regressor** | Gradient boosting untuk prediksi nilai kontinu |
| **LightGBM Regressor** | Gradient boosting cepat untuk dataset besar |

### Preprocessing
- IQR Clipping (1%–99%) untuk outlier pada target
- Median Imputation untuk missing values
- StandardScaler untuk normalisasi fitur

### Hyperparameter Tuning
- **Optuna** dengan 30 trials per model
- Optimasi metrik: **RMSE** via 5-Fold Cross-Validation

### Metrics Evaluasi
| Metric | Deskripsi |
|---|---|
| **MSE** | Mean Squared Error |
| **RMSE** | Root MSE — dalam satuan tahun |
| **MAE** | Mean Absolute Error |
| **R²** | Koefisien determinasi (0–1, semakin tinggi semakin baik) |

### Interpretasi Model
- **LIME** (Local Interpretable Model-agnostic Explanations) digunakan untuk menjelaskan prediksi model pada level individu sampel

### Tracking
- Semua eksperimen dilacak dengan **MLflow**

---

## ⚙️ Cara Menjalankan

### Menggunakan Google Colab (Rekomendasi)

1. Buka [Google Colab](https://colab.research.google.com)
2. Upload notebook `.ipynb` yang sesuai
3. Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
4. Jalankan setiap cell secara berurutan dari atas ke bawah
5. Upload dataset saat diminta

### Instalasi Library

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn \
            xgboost lightgbm optuna mlflow lime
```

---

## 📊 Ringkasan Hasil

### Task 1 — Fraud Detection
| Model | AUC-ROC | F1-Score | Precision | Recall |
|---|---|---|---|---|
| XGBoost | *lihat notebook* | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| LightGBM | *lihat notebook* | *lihat notebook* | *lihat notebook* | *lihat notebook* |

### Task 2 — Regression
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| XGBoost | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| LightGBM | *lihat notebook* | *lihat notebook* | *lihat notebook* |

> 📌 Hasil lengkap tersedia di masing-masing notebook setelah dijalankan.

---

## 🛠️ Tech Stack

| Library | Versi | Kegunaan |
|---|---|---|
| Python | 3.10+ | Bahasa pemrograman utama |
| NumPy & Pandas | Latest | Manipulasi data |
| Scikit-learn | 1.3+ | Preprocessing & metrics |
| XGBoost | 2.0+ | Model gradient boosting |
| LightGBM | 4.0+ | Model gradient boosting |
| Optuna | 3.4+ | Hyperparameter tuning |
| MLflow | 2.9+ | Experiment tracking |
| LIME | Latest | Model interpretability |
| Imbalanced-learn | 0.11+ | SMOTE oversampling |
| Matplotlib & Seaborn | Latest | Visualisasi |

---

*Muhammad Aqsandy J Iskandar — TK-46-02 — 1103220214*
