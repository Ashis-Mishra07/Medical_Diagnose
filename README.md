# 🏥 Multiple Disease Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![ML Models](https://img.shields.io/badge/ML%20Models-Trained-green.svg)](#)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)](#)

A machine learning-powered web application built with Streamlit to predict the presence of common diseases including Diabetes, Heart Disease, Parkinson’s, and Breast Cancer. Users can enter basic health metrics and receive immediate predictions.

---

## 🎯 Project Overview

This project aims to provide a simple yet effective tool for early disease prediction. It features:

- ✅ Trained ML models saved as `.sav` files
- 🎛️ Interactive web interface using Streamlit
- 📦 Easy-to-install Python environment
- 🔍 Real-time disease prediction

---

## 🧠 Supported Diseases

The following diseases are currently supported:

- 🔷 **Diabetes**
- 🔴 **Heart Disease**
- 🟠 **Parkinson’s Disease**
- 🟢 **Breast Cancer**
- 🔷 **Get Diagnosed**
  
---

## 🚀 Quick Start

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Ashis-Mishra07/Medical_Diagnose.git
cd Medical_Diagnose

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

The app will open at **http://localhost:8501**


## 📁 Project Structure

```
Multiple_Disease_Prediction_Model/
├── app.py                           # Main Streamlit web application
├── requirements.txt                 # List of all required Python libraries
├── models/                          # Directory containing trained models
│   ├── diabetes_model.sav               # Machine Learning model (Pickle format)
│   ├── heart_disease_model.sav          # Machine Learning model (Pickle format)
│   ├── parkinsons_model.sav             # Machine Learning model (Pickle format)
│   ├── breast_cancer_model.keras        # Deep Learning model (TensorFlow Keras format)
│   └── diagnosis_model.sav              # Additional ML model for general diagnosis
└── README.md                        # Project documentation (you’re reading it!)

```


---

## 🧬 Advanced Diagnosis Model

An enhanced machine learning-powered module that not only **predicts the disease** but also provides a **comprehensive health recommendation package**.

### 🌟 What You’ll Get:

- 🦠 **Disease**  
  Get a prediction of the most probable disease based on your inputs.

- 📖 **Description**  
  A clear and concise explanation of the diagnosed condition.

- 🛡️ **Precaution**  
  Proactive steps to reduce risk or manage the condition effectively.

- 💊 **Medication**  
  A list of commonly prescribed medicines for reference *(not for self-medication)*.

- 🏃 **Workout**  
  Recommended physical activities and fitness routines that support your condition.

- 🥗 **Diets**  
  Nutrition advice and diet plans to assist in managing or preventing the disease.

This feature aims to **bridge prediction with actionable lifestyle insights**, helping users make informed, health-conscious decisions post-diagnosis.

---





## 🔮 Future Enhancements

### Planned Features
- [ ] Add model accuracy indicators
- [ ] Add model retraining support from UI
- [ ] Integrate user authentication
- [ ] Integration with EEG devices
- [ ] Deploy on Streamlit Cloud or HuggingFace Spaces


---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

**📅 Last Updated**: June 2025 | **🔢 Version**: 1.0.0
