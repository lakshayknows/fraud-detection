# 🔍 Fraud Detection Model & Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Model Accuracy](https://img.shields.io/badge/XGBoost-99.95%25-brightgreen)
![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit-orange)

## 📌 Project Overview  
This project tackles the challenge of identifying **fraudulent transactions** using machine learning. It includes:
- Model training & evaluation in `internship.ipynb`
- A Streamlit dashboard for real-time predictions 🎛️

---

## 📊 Dataset  
📂 The dataset is hosted on Kaggle:  
👉 [Kaggle Dataset – Fraud Detection](https://www.kaggle.com/datasets/lakshaykahai/fraud-detection)

Due to size constraints, it’s not uploaded to this repository.

---

## 🧹 Preprocessing Steps
- **Cleaned** missing values  
- **Encoded** categorical data  
- **Scaled** numerical values using MinMaxScaler  
- **Split** into training and testing sets  

---

## 🤖 Models & Performance

| Model                | Accuracy   |
|---------------------|------------|
| Logistic Regression | 99.87%     |
| Decision Tree       | 99.95%     |
| XGBoost             | 99.95%     |

📈 *Evaluation metrics used:* Accuracy, Precision, Recall, F1-Score

---

## 🚀 Streamlit App Deployment

The app is deployed using Streamlit and allows real-time fraud prediction. Try it here:  
🌐 **[Live App – Fraud Detection Dashboard](https://fraud-detection-lakshay.streamlit.app/)**

### 🧰 To run locally:
1. Clone the repo  
2. Download dataset from [Kaggle](https://www.kaggle.com/datasets/lakshaykahai/fraud-detection) and place it in the project root  
3. Run:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
