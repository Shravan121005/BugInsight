# 🐞 BugInsight - GitHub Issue Severity & Resolution Time Prediction
uvicorn api.main:app 

{
  "summary": "VS Code crashes while opening terminal",
  "description": "The application crashes whenever a new terminal is created.",
  "component_name": "Terminal",
  "product_name": "Firefox",
  "quantity_of_votes": 12,
  "quantity_of_comments": 5
}

BugInsight is an end-to-end Machine Learning application that predicts the **severity** of a software bug and estimates its **resolution time** from a bug report.

The project combines Natural Language Processing (NLP), Machine Learning, and FastAPI to provide real-time predictions through a REST API.

---

## 🚀 Features

- Predicts bug severity
- Estimates bug resolution time (in days)
- REST API built using FastAPI
- Interactive Swagger UI for testing
- TF-IDF based text feature extraction
- XGBoost models for classification and regression

---

## 🏗️ Project Architecture

```
Bug Report
      │
      ▼
Data Cleaning
      │
      ▼
TF-IDF + OneHot Encoding
      │
      ▼
Severity Classifier (XGBoost)
      │
      ▼
Resolution Time Regressor (XGBoost)
      │
      ▼
FastAPI
      │
      ▼
JSON Response
```

---

## 📂 Project Structure

```
BugInsight/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── preprocessor.pkl
│   ├── severity_model.pkl
│   ├── resolution_model.pkl
│   ├── dataset.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_classifier.py
│   └── train_regressor.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy

### NLP

- TF-IDF Vectorizer

### Backend

- FastAPI
- Uvicorn

### Model Serialization

- Joblib

---

## 📊 Dataset - Long Lived Bug Prediction(Kaggle)

The model is trained on a real-world software bug dataset containing:

- Bug Summary
- Bug Description
- Product Name
- Component Name
- Number of Votes
- Number of Comments
- Severity Category
- Bug Fix Time

---

## ⚙️ Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Text concatenation
- Feature engineering

### Feature Extraction

- TF-IDF Vectorization
- One-Hot Encoding
- Numerical Feature Integration

### Models

#### Severity Prediction

- XGBoost Classifier

#### Resolution Time Prediction

- XGBoost Regressor

---

## 🚀 Running the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/BugInsight.git

cd BugInsight
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python src/feature_engineering.py

python src/train_classifier.py

python src/train_regressor.py
```

### Start API

```bash
uvicorn api.main:app --reload
```

---

## 📌 API Documentation

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Sample Request

```json
{
  "summary": "VS Code crashes while opening terminal",
  "description": "The application crashes whenever a new terminal is created.",
  "component_name": "Terminal",
  "product_name": "Firefox",
  "quantity_of_votes": 12,
  "quantity_of_comments": 5
}
```

---

## Sample Response

```json
{
  "severity": "normal",
  "estimated_fix_time": 30.51
}
```

---

## Future Improvements

- Handle class imbalance
- Hyperparameter tuning
- Advanced NLP using BERT/Sentence Transformers
- Docker deployment
- CI/CD with GitHub Actions
- Cloud deployment (AWS/GCP)

---

## Author

**Shravan Jain**
