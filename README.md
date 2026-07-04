# ❤️ Heart Disease Prediction

A Machine Learning project that predicts whether a patient is likely to have heart disease using the UCI Heart Disease dataset.

---

## 📌 Project Overview

This project uses the **Logistic Regression** algorithm to classify whether a patient has heart disease based on medical attributes.

The dataset is cleaned, preprocessed, and used to train a machine learning model that can predict heart disease for new patients.

---

## 🚀 Features

- Load UCI Heart Disease dataset
- Data preprocessing
- Handle missing values
- Feature engineering
- Train Logistic Regression model
- Evaluate model accuracy
- Save trained model using Joblib
- Predict heart disease using user input

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

## 📂 Project Structure

```
Heart-Disease-Prediction/
│
├── dataset/
│   └── heart.csv
│
├── models/
│   └── heart_model.pkl (generated after training)
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

Dataset used:

**UCI Heart Disease Dataset**

https://archive.ics.uci.edu/ml/datasets/Heart+Disease

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/krj250907/Heart-Disease-Prediction.git
```

Go inside the project

```bash
cd Heart-Disease-Prediction
```

Create virtual environment

```bash
python3 -m venv venv
```

Activate virtual environment

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python3 src/train.py
```

Output

```
Model Accuracy: 88.33%

Heart Disease Prediction Model Saved Successfully!
```

---

## ▶️ Predict

```bash
python3 src/predict.py
```

Example

```
Age: 52
Sex: 1
Chest Pain Type: 2
...
```

Output

```
✅ No Heart Disease Detected
```

or

```
⚠️ Heart Disease Detected
```

---

## 🤖 Machine Learning Algorithm

- Logistic Regression

---

## 📈 Model Accuracy

```
88.33%
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Kartik Raj**

GitHub:
https://github.com/krj250907