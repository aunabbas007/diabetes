# Diabetes Prediction using KNN 🩺

## 📌 Project Overview
This project predicts whether a patient has diabetes using the K-Nearest Neighbors (KNN) algorithm.

## 📊 Dataset
- Pima Indians Diabetes Dataset
- Features include Glucose, BMI, Age, Insulin, etc.

## ⚙️ Steps Performed
1. Data Cleaning
   - Replaced invalid zero values with NaN
   - Filled missing values using median

2. Exploratory Data Analysis (EDA)
   - Distribution plots
   - Feature vs outcome analysis
   - Correlation heatmap

3. Data Preprocessing
   - Feature scaling using StandardScaler

4. Model Building
   - Implemented KNN classifier
   - Tuned K value from 1 to 20

5. Evaluation
   - Accuracy score
   - Confusion matrix
   - Classification report

## 🧠 Model Used
- K-Nearest Neighbors (KNN)

## 📈 Results
- Best K value: (auto-selected)
- Accuracy: ~70–80%

## 🔍 Key Insights
- Glucose and BMI are strong indicators of diabetes
- Proper scaling is critical for KNN performance
- Choosing optimal K improves model accuracy

## 🚀 How to Run

```bash
pip install -r requirements.txt
python model.py