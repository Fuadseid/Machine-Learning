# Diabetes Prediction Project

This project aims to predict whether a person has diabetes based on various diagnostic features, such as age, BMI, blood pressure, glucose levels, etc. The project uses machine learning to build a classifier that can predict the likelihood of diabetes in a given individual.

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

## Project Overview
This project uses the Pima Indians Diabetes Dataset to build a machine learning model that predicts whether a patient has diabetes based on diagnostic factors. The dataset includes information like glucose levels, BMI, age, etc., and the model is trained using Logistic Regression, Random Forest, and SVM classifiers. The goal is to predict the Outcome variable, where 0 means no diabetes, and 1 means the patient has diabetes.

## Getting Started
Follow these steps to get the project up and running on your local machine.

### Step 1: Clone the Repository
First, clone the repository to your local machine.
```
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

### Step 2: Download the Dataset
Download the Pima Indians Diabetes Dataset from the following link:
- **Dataset:** diabetes.csv

Ensure the dataset is placed in the project folder before proceeding.

## Prerequisites
Ensure you have Python 3.6+ installed on your machine. You can check your Python version by running:
```
python --version
```
You will also need to install the necessary Python libraries to run the project.

## Installing Dependencies
Create a virtual environment (optional, but recommended) and activate it:

**On Windows:**
```
python -m venv env
.\env\Scripts\activate
```

**On Mac/Linux:**
```
python3 -m venv env
source env/bin/activate
```

Then install the required dependencies:
```
pip install -r requirements.txt
```
Alternatively, you can install the necessary libraries individually using pip:
```
pip install pandas numpy scikit-learn seaborn matplotlib fastapi uvicorn
```

## Running the Project

### 1. Run the Jupyter Notebook
You can run the project directly in a Jupyter notebook. If you don’t have Jupyter installed, install it with:
```
pip install notebook
```
Then launch Jupyter Notebook:
```
jupyter notebook
```
Open the notebook `diabetes_prediction.ipynb` and follow the steps to load the dataset, preprocess the data, train the model, and evaluate its performance.

### 2. Run the API (Optional)
To deploy the trained model as an API using FastAPI, follow these steps:
- Make sure the model is saved (using joblib or pickle).
- Run the FastAPI server:
```
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000/`. You can test it by sending a POST request with patient data using tools like Postman or cURL.

## Data Description
The dataset contains the following columns:
1. **Pregnancies**: Number of times the patient has been pregnant.
2. **Glucose**: Plasma glucose concentration in an oral glucose tolerance test.
3. **BloodPressure**: Diastolic blood pressure (mm Hg).
4. **SkinThickness**: Triceps skinfold thickness (mm).
5. **Insulin**: 2-Hour serum insulin (mu U/ml).
6. **BMI**: Body mass index (weight in kg/(height in m)^2).
7. **DiabetesPedigreeFunction**: A function that represents the genetic relationship to diabetes.
8. **Age**: Age of the patient.
9. **Outcome**: Target variable (1 = Has diabetes, 0 = No diabetes).

## Model Explanation
The project uses several machine learning models to predict diabetes:
1. **Logistic Regression**: A basic linear classifier, suitable for binary classification tasks.
2. **Random Forest Classifier**: An ensemble method that builds multiple decision trees and merges their results.
3. **Support Vector Machine (SVM)**: A classifier that finds the optimal hyperplane to separate classes.

You can experiment with different models to find the best-performing one for this dataset.

## Evaluation Metrics
We use several metrics to evaluate the model’s performance:
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positive cases that are correctly identified.
- **F1-score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A table used to describe the performance of a classification model.

## Deployment
The model has been deployed on **Render Cloud**, and you can now access it via the following link:
[Diabetes Prediction API](https://machine-learning-1-yl7v.onrender.com/predict)

### How to Use the Cloud-Deployed API
You can test the API by sending a POST request with patient data. Below are the steps to test it using **Postman** or any similar tool:

#### Using Postman:
1. Open Postman and create a **new request**.
2. Set the request type to **POST**.
3. Enter the API endpoint:
   ```
   https://machine-learning-1-yl7v.onrender.com/predict
   ```
4. In the **Body** section, select **raw** and set the type to **JSON**.
5. Enter the following JSON data:
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
6. Click **Send**.
7. The API will return a response with the prediction (0 for no diabetes, 1 for diabetes).

#### Using cURL:
Alternatively, you can use cURL in your terminal:
```
curl -X POST "https://machine-learning-1-yl7v.onrender.com/predict" -H "Content-Type: application/json" -d '{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 90,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 32
}'
```

## Limitations and Future Improvements
### Limitations:
- **Data Imbalance**: The dataset contains more non-diabetic cases than diabetic cases, which may impact model performance.
- **Feature Constraints**: The dataset does not include lifestyle habits, medication, or detailed family medical history.

### Future Improvements:
- Experiment with class imbalance techniques (e.g., SMOTE, undersampling).
- Expand the dataset by incorporating additional features.
- Experiment with advanced models such as XGBoost or Neural Networks to improve accuracy.

## Acknowledgements
- **Pima Indians Diabetes Dataset** – Original dataset source.
- **FastAPI** – Framework used for deploying the model as an API.

