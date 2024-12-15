import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Fungsi tambahan
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def train_model(X_train, y_train, n_estimators):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Streamlit UI
st.title("Diabetes Detection with Random Forest")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your diabetes dataset (CSV format)", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Step 2: Data Cleaning
    st.subheader("Data Cleaning")
    data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    ].replace(0, np.NaN)

    st.write("Missing values (before cleaning):")
    st.write(data.isnull().sum())

    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        data[col].fillna(data[col].mean(), inplace=True)

    st.write("Missing values (after cleaning):")
    st.write(data.isnull().sum())

    # Step 3: Visualize Correlation Matrix
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cbar=False, cmap="icefire")
    st.pyplot(plt)

    # Step 4: Data Balancing
    st.subheader("Data Balancing")
    data_major = data[data['Outcome'] == 0]
    data_minor = data[data['Outcome'] == 1]
    upsample = resample(data_minor, replace=True, n_samples=len(data_major), random_state=42)
    balanced_data = pd.concat([data_major, upsample])

    st.write("Class distribution after balancing:")
    st.write(balanced_data['Outcome'].value_counts())

    # Step 5: Train-Test Split
    st.subheader("Train-Test Split")
    X = balanced_data.drop('Outcome', axis=1)
    y = balanced_data['Outcome']
    test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    st.write(f"Training data size: {len(X_train)}")
    st.write(f"Testing data size: {len(X_test)}")

    # Step 6: Train Random Forest Model
    st.subheader("Train Random Forest Model")
    n_estimators = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=200, value=100, step=10)
    model, scaler = train_model(X_train, y_train, n_estimators)

    # Step 7: Model Evaluation
    st.subheader("Model Evaluation")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = (y_test == y_pred).mean()
    precision = (confusion_matrix(y_test, y_pred)[1, 1]) / (confusion_matrix(y_test, y_pred)[0, 1] + confusion_matrix(y_test, y_pred)[1, 1])
    recall = (confusion_matrix(y_test, y_pred)[1, 1]) / (confusion_matrix(y_test, y_pred)[1, 0] + confusion_matrix(y_test, y_pred)[1, 1])
    f1 = 2 * (precision * recall) / (precision + recall)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=model.classes_)
    disp.plot(cmap="Blues", values_format="d")
    st.pyplot(disp.figure_)

    # Step 8: Predict with User Input
    st.subheader("Predict Diabetes Based on Input")
    st.write("Enter the following values for prediction:")

    # Input fields for user
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0)
    insulin = st.number_input("Insulin", min_value=0.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, step=0.01)
    age = st.number_input("Age", min_value=0, step=1)

    # Collect input values into a DataFrame
    user_input = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Predict button
    if st.button("Predict"):
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        probability = prediction_proba[0][prediction[0]] * 100
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {probability:.2f}%")

else:
    st.info("Please upload a CSV file to proceed.")
