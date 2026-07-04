import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/heart_model.pkl")

print("=" * 40)
print("❤️ Heart Disease Prediction System")
print("=" * 40)

# Take User Input
age = float(input("Age: "))
sex = float(input("Sex (1=Male, 0=Female): "))
cp = float(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar (>120mg/dl: 1 else 0): "))
restecg = float(input("Rest ECG (0-2): "))
thalach = float(input("Maximum Heart Rate Achieved: "))
exang = float(input("Exercise Induced Angina (1=Yes, 0=No): "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("Slope (0-2): "))
ca = float(input("Number of Major Vessels (0-3): "))
thal = float(input("Thal (1=Normal, 2=Fixed, 3=Reversible): "))

# Create DataFrame with feature names
data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
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

# Predict
prediction = model.predict(data)

print("\n" + "=" * 40)
print("Prediction Result")
print("=" * 40)

if prediction[0] == 1:
    print("⚠️ Heart Disease Detected")
    print("Recommendation: Please consult a cardiologist for further evaluation.")
else:
    print("✅ No Heart Disease Detected")
    print("Recommendation: Continue maintaining a healthy lifestyle.")

print("=" * 40)