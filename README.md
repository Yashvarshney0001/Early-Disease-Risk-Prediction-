# Early Disease Risk Prediction

A machine learning-powered web app for predicting the likelihood of diabetes using patient health data. Built using Python, Streamlit, and Scikit-learn.

---

# Repository Structure

├── app.py                  # Streamlit frontend for the prediction
├── diabetes.csv            # Sample dataset (PIMA Indians Diabetes Database)
├── feature_importance.png  # Feature importance graph from model
├── project.py              # Alternative script / experimentation
├── save.py                 # Utility for saving model
├── rf_model.pkl            # Trained Random Forest model
├── scaler.pkl              # Saved standard scaler object
├── README.md               # Project documentation

# Demo
Want to try it out? Clone the repo and run the app:

git clone https://github.com/Yashvarshney0001/Early-Disease-Risk-Prediction.git
cd Early-Disease-Risk-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
# Dataset Used
PIMA Indian Diabetes Dataset

Source: Kaggle

Features include:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

# Features
 Real-time disease prediction

 Interactive web UI using Streamlit

 Model trained on real health data

 Visualization of feature importance

# Tech Stack
Python

Pandas, NumPy

Scikit-learn (Random Forest Classifier)

Streamlit (Web UI)

Joblib (for model serialization)


