

# Diabetes Prediction Project

This project predicts whether an individual has diabetes based on various diagnostic features such as age, BMI, blood pressure, and glucose levels. The model is built using **K-Nearest Neighbors (KNN)** and is deployed with a frontend interface for better user interaction.

## Table of Contents
1. Project Overview  
2. Getting Started  
3. Prerequisites  
4. Installing Dependencies  
5. Running the Project  
6. Data Description  
7. Model Explanation  
8. Evaluation Metrics  
9. Deployment  
10. Limitations and Future Improvements  
11. Acknowledgements  

---

## Project Overview
The project utilizes the **Pima Indians Diabetes Dataset** to train a **K-Nearest Neighbors (KNN)** model for predicting diabetes. The target variable (Outcome) is binary, where **1 indicates diabetes** and **0 indicates no diabetes**.  

To enhance accessibility, a **React-based frontend** has been developed, enabling users to interact easily with the model.

---

## Getting Started

### Step 1: Clone the Repository
```
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

### Step 2: Download the Dataset
- Dataset: **diabetes.csv**  
Ensure the dataset is placed in the project directory before proceeding.

---

## Prerequisites
Ensure Python **3.6+** is installed:
```
python --version
```

Required Python libraries:
```
pip install pandas numpy scikit-learn seaborn matplotlib fastapi uvicorn
```

---

## Running the Project

### 1. Run Jupyter Notebook (For EDA & Model Training)
```
pip install notebook
jupyter notebook
```
Open `diabetes_prediction.ipynb` to execute model training and evaluation.

### 2. Run the API (For Deployment)
Run the FastAPI server:
```
uvicorn main:app --reload
```
API available at: `http://127.0.0.1:8000/`

---

## Data Description

The dataset consists of **768 instances** with the following features:

- **Pregnancies**: Number of times pregnant  
- **Glucose**: Plasma glucose concentration  
- **BloodPressure**: Diastolic blood pressure (mm Hg)  
- **SkinThickness**: Triceps skinfold thickness (mm)  
- **Insulin**: 2-hour serum insulin (µU/mL)  
- **BMI**: Body mass index  
- **DiabetesPedigreeFunction**: Diabetes likelihood based on family history  
- **Age**: Age of the individual  
- **Outcome**: **1 = Diabetes, 0 = No diabetes**  

---

## Model Explanation

The project uses **K-Nearest Neighbors (KNN)**:
- Determines a sample's class based on the **majority vote of its K nearest neighbors**.  
- Uses **Euclidean distance** for measuring proximity.  
- Users can experiment with different values of **K** for optimal performance.

---

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness  
- **Precision**: True positive rate among predicted positives  
- **Recall**: Percentage of actual diabetics correctly identified  
- **F1-score**: Harmonic mean of precision & recall  
- **Confusion Matrix**: Breakdown of correct and incorrect classifications  

---

## Deployment  

### Cloud Deployment
The model is deployed on **Render Cloud** with a public API:  
API documentation:  
[https://machine-learning-1-yl7v.onrender.com/docs](https://machine-learning-1-yl7v.onrender.com/docs)

### API Usage
#### Using Postman:
1. **POST** request to:
   ```
   https://machine-learning-1-yl7v.onrender.com/predict
   ```
2. **Request Body (JSON format):**
   ```json
   {
     "Pregnancies": 2,
     "Glucose": 120,
     "BloodPressure": 70,
     "SkinThickness": 25,
     "Insulin": 90,
     "BMI": 28.5,
     "DiabetesPedigreeFunction": 0.5,
     "Age": 32
   }
   ```
3. **Response:** `{"prediction": 1}` (1 = Diabetes, 0 = No diabetes)

#### Frontend Application
A **React-based web app** has been developed for user-friendly interaction with the API. The app can be accessed at:  
[https://ml-t4w1.vercel.app/](https://ml-t4w1.vercel.app/)

---

## Limitations and Future Improvements

### Limitations:
- **Render Cloud Free Tier Delay**: On first request, the API may be slow and show a fetching error. Pressing "Predict" 4-5 times resolves this issue. The issue is due to Render’s free plan limitations; upgrading to a paid version would fix it.  
- **Data Imbalance**: More non-diabetic cases than diabetic cases, possibly affecting model bias.  
- **Limited Features**: Does not include lifestyle factors like diet and physical activity.

### Future Improvements:
- Implement **SMOTE (Synthetic Minority Over-sampling)** to balance classes.  
- Extend dataset with additional health indicators.  
- Explore more advanced models like **XGBoost** and **Neural Networks**.  

---

## Acknowledgements  
- **Pima Indians Diabetes Dataset** – Data source.  
- **FastAPI** – Framework for deploying the machine learning model.  

