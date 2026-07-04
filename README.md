# 🎯 Criteo Ad Click-Through Rate (CTR) Prediction

[![Open Phase 2 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SowmyaGooty/criteo-ad-click-prediction/blob/main/phase2.ipynb)

## 📌 Overview

A large-scale distributed machine learning project predicting **Ad Click-Through Rates (CTR)** on the **Criteo Private Ad Dataset** — one of the largest publicly available advertising datasets. The project processes **100M+ ad impressions (~34 GB)** using **Apache Spark** across all 30 days of data, implementing a full end-to-end ML pipeline from data loading to model evaluation.

---

## 📊 Dataset

- **Source:** [Criteo Private Ad Dataset](https://huggingface.co/datasets/criteo/CriteoPrivateAd) — Hugging Face
- **Size:** ~34 GB | 100M+ ad impression records | 30 days
- **Key columns:** `id`, `user_id`, `campaign_id`, `day_int`, `click` (target)
- **Challenge:** 95–99% non-clicks (severe class imbalance)

---

## 🔬 Project Structure

### 📁 Phase 2 — Data Preparation & Exploration (`phase2.ipynb`)

| Step | Description |
|---|---|
| **1. Dataset Loading** | Downloaded full 34 GB dataset from Hugging Face using `snapshot_download` |
| **2. Data Cleaning** | Removed duplicates, null core identifiers; lazy transformation chain |
| **3. SQL Queries** | Spark SQL exploration — click rates by campaign, day trends, user activity |
| **4. Statistical Analysis** | Distribution analysis, null audits via 1% sampling |
| **5. Discussion** | Memory strategy, schema analysis, data quality findings |

### 📁 Phase 3 — Feature Engineering & ML Pipeline (`phase3.ipynb`)

| Step | Description |
|---|---|
| **1. Environment Setup** | Spark session tuned for 24 GB RAM with vectorized reader disabled |
| **2. Feature Engineering** | User historical CTR, log-transformed display order, campaign/user impression frequency |
| **3. Broadcast Joins** | Replaced shuffle joins for small lookup tables — dramatic speedup on 100M rows |
| **4. ML Pipeline** | StringIndexer → Imputer → VectorAssembler → StandardScaler |
| **5. Model Training** | Logistic Regression + Random Forest classifiers |
| **6. Evaluation** | AUC-ROC, Accuracy, Feature Importance analysis |

---

## 📈 Results

| Model | AUC-ROC | Accuracy |
|---|---|---|
| **Logistic Regression** | **0.99** | **~94.6%** |
| **Random Forest** | **0.99** | **~94.6%** |

### 🔑 Key Finding
`user_hist_ctr` was the **dominant feature** with **0.976 Gini importance** in Random Forest — confirming that **behavioral signals are the strongest CTR predictor**.

---

## ⚙️ Memory Engineering Highlights

| Challenge | Solution |
|---|---|
| 34 GB OOM crashes | Disabled vectorized Parquet reader |
| Full dataset scan on `.count()` | Omitted premature `.count()` calls |
| Slow shuffle joins | Used Spark `broadcast()` for small lookup tables |
| Memory overflow during training | Write-to-disk instead of `.persist()` on raw DataFrame |
| 24 GB local machine limit | Tuned Spark: 16 GB driver, 8 GB executor |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**Libraries Used:**
- `pyspark` — Distributed data processing and ML
- `pyspark.sql.functions` — Data transformations
- `pyspark.ml` — ML pipeline (StringIndexer, VectorAssembler, StandardScaler)
- `pyspark.ml.classification` — LogisticRegression, RandomForestClassifier
- `pyspark.ml.evaluation` — BinaryClassificationEvaluator
- `huggingface_hub` — Dataset download
- `pyarrow` — Parquet file handling

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Java 8 or 11 (required for Spark)
- 24 GB+ RAM recommended
- ~40 GB free disk space

### Installation
```bash
pip install pyspark huggingface_hub pyarrow pandas
```

### Run Phase 2
```bash
jupyter notebook phase2.ipynb
```

### Run Phase 3
```bash
jupyter notebook phase3.ipynb
```

> ⚠️ **Note:** The dataset is ~34 GB. Download may take 1–2 hours depending on connection speed. The notebook is configured to be resumable if interrupted.

---

## 📁 Project Structure

```
criteo-ad-click-prediction/
│
├── phase2.ipynb     # Data loading, cleaning, SQL exploration
├── phase3.ipynb     # Feature engineering + ML pipeline (all 30 days)
└── README.md        # Project documentation
```

---

## 💡 Key Engineering Decisions

- **Write-to-disk strategy** instead of `.persist()` on 34 GB raw DataFrame
- **Lazy transformation chains** — no premature actions that trigger full scans
- **Broadcast joins** for small lookup tables to avoid expensive shuffle operations
- **Median imputation** for sparse nulls without distributional bias
- **StandardScaler** to normalize feature scales for Logistic Regression convergence
- **AUC-ROC as primary metric** due to 95–99% class imbalance in CTR data

---

## 👩‍💻 Author

**Sowmya Gooty**
MS Engineering Data Science @ University of Houston
📧 sgooty@cougarnet.uh.edu
🔗 [LinkedIn](https://linkedin.com/in/sowmyagooty) | [GitHub](https://github.com/SowmyaGooty)

---

⭐ If you found this project useful, please consider giving it a star!
