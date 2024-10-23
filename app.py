import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Jane-Pereira/Health-Risks-during-Pregnancy/main/Maternal%20Health%20Risk%20Data%20Set.csv")

# Load the dataset
dataset = load_data()

# Inspect dataset
st.title('Pregnancy Complication Risk Prediction')
st.write('This app predicts the risk level of pregnancy complications based on patient data.')

# Show unique risk levels
st.write("Unique Risk Levels:", dataset['RiskLevel'].unique())
st.write(dataset.info())

# Features and labels
X = dataset[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].values
y = dataset['RiskLevel'].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train RandomForestClassifier
rcla = RandomForestClassifier(n_estimators=500)
rcla.fit(X_train, y_train)

# Predict on test set
rpred = rcla.predict(X_test)

# Evaluate the model
rac = accuracy_score(y_test, rpred)
st.write(f"Model Accuracy: {rac:.2f}")

# Show confusion matrix if checkbox is selected
if st.checkbox('Show Confusion Matrix'):
    cm = confusion_matrix(y_test, rpred)
    st.write("Confusion Matrix:")
    st.write(cm)

# User input for patient details
st.subheader('Enter Patient Report Details:')
age = st.number_input('Age', min_value=15, max_value=100, value=25)
systolic_bp = st.number_input('Systolic BP', min_value=70, max_value=180, value=120)
diastolic_bp = st.number_input('Diastolic BP', min_value=40, max_value=120, value=80)
bs = st.number_input('Blood Sugar (BS)', min_value=50.0, max_value=250.0, value=100.0)
body_temp = st.number_input('Body Temperature (BodyTemp)', min_value=25.0, max_value=45.0, value=36.5)
heart_rate = st.number_input('Heart Rate', min_value=50, max_value=180, value=80)

# Prepare input data for prediction
input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]])

# Predict risk level
if st.button('Predict Risk Level'):
    prediction = rcla.predict(input_data)
    st.subheader('Risk Level Prediction')
    st.write(f'Possible Stages: Low, Medium, High')
    st.write(f'Predicted Risk Level: {prediction[0]}')
