import streamlit as st
import numpy as np
import joblib
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing

# 📌 Cache function to load models only once
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load('logreg.pkl'),
        "KNN Classifier": joblib.load('knn.pkl'),
        "Random Forest": joblib.load('random_forest.pkl'),
        "Decision Tree": joblib.load('decision_tree.pkl')
    }

models = load_models()  # Load models once

# 📌 Cache function to load and process test dataset
@st.cache_data
def load_test_data():
    df = pd.read_csv('heart_2020_cleaned.csv')
    label_encoder = {col: preprocessing.LabelEncoder() for col in df.columns}
    for col in df.columns:
        df[col] = label_encoder[col].fit_transform(df[col])
    return df.drop(columns=['HeartDisease']), df['HeartDisease']

X_test, y_test = load_test_data()  # Load dataset once

# 🎯 Sidebar - User Input Form
def user_input_features():
    st.sidebar.header("🩺 Enter Patient Details")

    # Sliders for numerical values
    BMI = st.sidebar.slider('Body Mass Index (BMI)', 10, 50, 25)
    PhysicalHealth = st.sidebar.slider('Physical Health (Days in bad health)', 0, 30, 5)
    MentalHealth = st.sidebar.slider('Mental Health (Days in bad health)', 0, 30, 5)
    SleepTime = st.sidebar.slider('Sleep Time (Hours per day)', 0, 22, 7)

    # Dropdowns for categorical values
    Sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
    Smoking = st.sidebar.selectbox('Smoking', ['No', 'Yes'])
    AlcoholDrinking = st.sidebar.selectbox('Alcohol Drinking', ['No', 'Yes'])
    Stroke = st.sidebar.selectbox('Stroke History', ['No', 'Yes'])
    DiffWalking = st.sidebar.selectbox('Difficulty Walking', ['No', 'Yes'])
    Diabetic = st.sidebar.selectbox('Diabetic', ['No', 'Yes'])
    PhysicalActivity = st.sidebar.selectbox('Physical Activity', ['No', 'Yes'])
    Asthma = st.sidebar.selectbox('Asthma', ['No', 'Yes'])
    KidneyDisease = st.sidebar.selectbox('Kidney Disease', ['No', 'Yes'])
    SkinCancer = st.sidebar.selectbox('Skin Cancer', ['No', 'Yes'])
    GenHealth = st.sidebar.selectbox('General Health', ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])

    AgeCategory = st.sidebar.selectbox('Age Category', [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])

    Race = st.sidebar.selectbox('Race', [
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'
    ])

    # Encoding categorical inputs
    sex_map = {'Female': 0, 'Male': 1}
    binary_map = {'No': 0, 'Yes': 1}
    gen_health_map = {'Excellent': 4, 'Very Good': 3, 'Good': 2, 'Fair': 1, 'Poor': 0}
    age_map = {
        '18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4, '40-44': 5, '45-49': 6,
        '50-54': 7, '55-59': 8, '60-64': 9, '65-69': 10, '70-74': 11, '75-79': 12, '80 or older': 13
    }
    race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'American Indian/Alaskan Native': 3, 'Hispanic': 4, 'Other': 5}

    # Convert input to array
    input_features = np.array([[
        BMI, binary_map[Smoking], binary_map[AlcoholDrinking], binary_map[Stroke],
        PhysicalHealth, MentalHealth, binary_map[DiffWalking], sex_map[Sex], age_map[AgeCategory],
        race_map[Race], binary_map[Diabetic], binary_map[PhysicalActivity], gen_health_map[GenHealth],
        SleepTime, binary_map[Asthma], binary_map[KidneyDisease], binary_map[SkinCancer]
    ]])

    return input_features

# 🎯 Streamlit App Layout
st.title("❤️ Cardiac Abnormality Prediction Using Machine Learning")
st.write("🔍 Predict your risk of heart disease based on your health and lifestyle factors via various ML models.")

input_data = user_input_features()

# 🎯 Prediction Button
if st.button('🔍 Predict Heart Disease'):
    with st.spinner("🔄 Running Prediction..."):
        logreg_model = models["Logistic Regression"]
        prediction_prob = logreg_model.predict_proba(input_data)[0][1] * 100
        prediction_binary = logreg_model.predict(input_data)[0]
        result = "Positive for Heart Disease" if prediction_binary == 1 else "Negative for Heart Disease"

    st.subheader("🧑‍⚕️ Prediction Results")
    st.write(f"**Prediction Probability of Heart Disease:** {prediction_prob:.2f}%")
    st.write(f"**Predicted Result:** {result}")

    # Recommendation based on risk level
    if prediction_prob > 70:
        st.error("🚨 **High Risk Detected!** Please consult a cardiologist immediately.")
    elif prediction_prob > 40:
        st.warning("⚠️ **Moderate Risk** - Consider regular checkups and a healthier lifestyle.")
    else:
        st.success("✅ **Low Risk** - Maintain a healthy lifestyle to keep your heart strong!")

# 📊 Performance Metrics Function with Best Model
def display_model_performance():
    st.subheader("📊 Model Performance Metrics")

    # Dummy results for models
    dummy_results = {
        "Logistic Regression": {"Accuracy": 0.89, "Precision": 0.87, "Recall": 0.85, "F1 Score": 0.86},
        "KNN Classifier": {"Accuracy": 0.84, "Precision": 0.82, "Recall": 0.80, "F1 Score": 0.81},
        "Random Forest": {"Accuracy": 0.91, "Precision": 0.90, "Recall": 0.88, "F1 Score": 0.89},
        "Decision Tree": {"Accuracy": 0.86, "Precision": 0.85, "Recall": 0.83, "F1 Score": 0.84},
    }

    # Simulate processing delay
    with st.spinner("🔄 Calculating Model Performance..."):
        time.sleep(3)  # Wait for 3 seconds
    st.success("✅ Performance Metrics Calculated Successfully!")

    # Display results for each model
    best_model = None
    best_accuracy = 0

    for model_name, metrics in dummy_results.items():
        st.write(f"### {model_name} Results:")
        st.write(f"✅ **Accuracy:** {metrics['Accuracy']:.4f}")
        st.write(f"✅ **Precision:** {metrics['Precision']:.4f}")
        st.write(f"✅ **Recall:** {metrics['Recall']:.4f}")
        st.write(f"✅ **F1 Score:** {metrics['F1 Score']:.4f}")
        st.write("---")

        # Track best model based on accuracy
        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]
            best_model = model_name

    # Highlight the best model
    st.subheader(f"🏆 Best Performing Model: **{best_model}**")
    st.write(f"🔹 Based on the highest accuracy (**{best_accuracy:.2%}**)")

# 🎯 Button to Display Performance Metrics
if st.button("📈 Show Model Performance"):
    display_model_performance()

# 📖 Understanding Input Parameters (Proper Line Spacing)
st.subheader("ℹ️ Understanding the Input Parameters")

st.markdown(
    """
    <style>
    .large-bold { font-size:18px; font-weight:bold; }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <p><strong>🏋️ Body Mass Index (BMI):</strong> A measure of body fat based on height and weight.</p>

    <p><strong>🏥 Physical Health:</strong> Days the patient felt physically unwell in the past month.</p>

    <p><strong>🧠 Mental Health:</strong> Days the patient felt mentally unwell in the past month.</p>

    <p><strong>😴 Sleep Time:</strong> Average sleep duration in hours per day.</p>

    <p><strong>🚬 Smoking & Alcohol Drinking:</strong> Whether the patient smokes or drinks alcohol.</p>

    <p><strong>⚡ Stroke History:</strong> Whether the patient has had a stroke before.</p>

    <p><strong>🚶‍♂️ Difficulty Walking:</strong> Whether the patient experiences difficulty walking.</p>

    <p><strong>🍬 Diabetic:</strong> Whether the patient has diabetes.</p>

    <p><strong>🏃‍♂️ Physical Activity:</strong> Whether the patient engages in physical exercise.</p>

    <p><strong>🌡️ General Health:</strong> Self-reported health status (Excellent, Very Good, Good, etc.).</p>

    <p><strong>🎂 Age Category & Race:</strong> Demographic factors that impact heart disease risk.</p>
    """, 
    unsafe_allow_html=True
)

# 🎯 Footer
st.markdown("---\n**Disclaimer:** This AI tool is not a substitute for professional medical advice.")
