import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Column names for UCI Heart Disease Dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Load Dataset
df = pd.read_csv(
    "dataset/heart.csv",
    names=columns
)

print("First 5 Rows:\n")
print(df.head())

# Replace ? with NaN
df.replace("?", pd.NA, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Convert columns to numeric
df = df.apply(pd.to_numeric)

# Convert target:
# 0 = No Disease
# 1,2,3,4 = Disease
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

print("\nDataset Shape:", df.shape)

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")

# Save Model
joblib.dump(model, "models/heart_model.pkl")

print("\nHeart Disease Prediction Model Saved Successfully!")