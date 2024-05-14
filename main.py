import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset (for demonstration purposes)
def load_data():
    return pd.read_csv(r"E:\IIT Hyderabad\Python\Machine Learning\heartdisease\heart_disease_data.csv")  # Change the filename to match your dataset


# Preprocess the data
def preprocess_data(df):
    # Perform any necessary preprocessing steps (e.g., handle missing values, feature scaling, etc.)
    # For simplicity, let's assume the data is already preprocessed in this example
    return df


# Train the machine learning model
def train_model(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# Display the UI
def main():
    st.title("HeartRisk Predictor")

    # Load data
    data = load_data()
    st.subheader("Input Health Metrics")
    age = st.number_input("Age", min_value=0, max_value=150, value=30)
    gender = st.radio("Gender", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
    exang = st.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0,
                              value=0.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Preprocess data
    df = preprocess_data(data)

    # Train model
    model = train_model(df)

    # Make prediction
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if gender == "Male" else 0],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })
    prediction = model.predict(input_data)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 0:
        st.write("Low risk of heart disease.")
    else:
        st.write("High risk of heart disease.")


if __name__ == "__main__":
    main()
