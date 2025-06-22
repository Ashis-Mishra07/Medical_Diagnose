import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from tensorflow import keras

# Run the app with: streamlit run "F:/Complete ML/All_Projects/MLProject5/app.py"
# conda activate mlproject

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# 👉👉👉 ADD THIS LINE BELOW 👇👇👇
# Replace the existing heading with this enhanced version
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 15px; position: relative;">
    <h1 style='color: #4CAF50; margin: 0;'>Multiple Disease Prediction System</h1>
    <div class="chat-icon-container">
        <a href="https://ashiskumarmishra-ai-chat.hf.space/" target="_blank" style="text-decoration: none;">
            <div class="chat-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M6 12.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5ZM3 8.062C3 6.76 4.235 5.765 5.53 5.886a26.58 26.58 0 0 0 4.94 0C11.765 5.765 13 6.76 13 8.062v1.157a.933.933 0 0 1-.765.935c-.845.147-2.34.346-4.235.346-1.895 0-3.39-.2-4.235-.346A.933.933 0 0 1 3 9.219V8.062Zm4.542-.827a.25.25 0 0 0-.217.068l-.92.9a24.767 24.767 0 0 1-1.871-.183.25.25 0 0 0-.068.495c.55.076 1.232.149 2.02.193a.25.25 0 0 0 .189-.071l.754-.736.847 1.71a.25.25 0 0 0 .404.062l.932-.97a25.286 25.286 0 0 0 1.922-.188.25.25 0 0 0-.068-.495c-.538.074-1.207.145-1.98.189a.25.25 0 0 0-.166.076l-.754.785-.842-1.7a.25.25 0 0 0-.182-.135Z"/>
                    <path d="M8.5 1.866a1 1 0 1 0-1 0V3h-2A4.5 4.5 0 0 0 1 7.5V8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1v-.5A4.5 4.5 0 0 0 10.5 3h-2V1.866ZM14 7.5V13a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V7.5A3.5 3.5 0 0 1 5.5 4h5A3.5 3.5 0 0 1 14 7.5Z"/>
                </svg>
            </div>
        </a>
        <span class="orbit-text orbit-text-1">Chat with AI</span>
        <span class="orbit-text orbit-text-2">Click me!</span>
        <span class="orbit-text orbit-text-3">Get help</span>
        <span class="orbit-text orbit-text-4">Ask anything</span>
    </div>
</div>

<style>
    .chat-icon-container {
        position: relative;
        width: 60px;
        height: 60px;
    }
    
    .chat-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #4CAF50;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        color: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        animation: pulse-animation 2s infinite, pulse-scale 3s ease-in-out infinite;
        transition: all 0.3s;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 2;
    }
    
    .chat-icon:hover {
        transform: translate(-50%, -50%) scale(1.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .orbit-text {
        position: absolute;
        padding: 3px 8px;
        background-color: #4CAF50;
        color: white;
        font-size: 12px;
        font-weight: bold;
        border-radius: 10px;
        z-index: 1;
        opacity: 0;
        white-space: nowrap;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    .orbit-text-1 {
        animation: orbit1 8s linear infinite;
    }
    
    .orbit-text-2 {
        animation: orbit2 8s linear infinite;
    }
    
    .orbit-text-3 {
        animation: orbit3 8s linear infinite;
    }
    
    .orbit-text-4 {
        animation: orbit4 8s linear infinite;
    }
    
    @keyframes orbit1 {
        0%, 100% { opacity: 0; transform: translate(20px, -20px) scale(0.5); }
        25% { opacity: 1; transform: translate(50px, 0) scale(1); }
        50% { opacity: 0; transform: translate(20px, 20px) scale(0.5); }
    }
    
    @keyframes orbit2 {
        0%, 100% { opacity: 0; transform: translate(-20px, -20px) scale(0.5); }
        25%, 75% { opacity: 0; }
        50% { opacity: 1; transform: translate(-50px, 0) scale(1); }
    }
    
    @keyframes orbit3 {
        0% { opacity: 0; transform: translate(20px, 20px) scale(0.5); }
        25%, 75% { opacity: 0; }
        50% { opacity: 0; }
        75% { opacity: 1; transform: translate(0, 50px) scale(1); }
        100% { opacity: 0; transform: translate(20px, 20px) scale(0.5); }
    }
    
    @keyframes orbit4 {
        0% { opacity: 0; transform: translate(-20px, 20px) scale(0.5); }
        25% { opacity: 1; transform: translate(0, -50px) scale(1); }
        50%, 100% { opacity: 0; }
    }
            
    @keyframes pulse-scale {
        0% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.5); }
        100% { transform: translate(-50%, -50%) scale(1); }
    }
    
    @keyframes pulse-animation {
        0% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Sidebar width and position */
    section[data-testid="stSidebar"] {
        width: 400px !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Remove outer padding/margin from all sidebar containers */
    [data-testid="stSidebar"] > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Try to target Streamlit's auto-generated sidebar container classes */
    div[data-testid="stSidebar"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    /* Remove top padding from the main block container as well */
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0 !important;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #f0f0f0;
        color: black;
        border-radius: 8px;
        caret-color: #222 !important;
    }

    /* Placeholder styling */
    .stTextInput > div > div > input::placeholder {
        color: #888 !important;
        opacity: 1 !important;
    }

    /* Help text styling */
    .css-1b0udgb {
        color: black !important;
        font-size: 0.85rem;
    }

    /* Stylish buttons */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        visibility: hidden;
        display: none;
    }
            
    .success-box {
        background-color: #06290e;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #28a745;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
            
</style>
""", unsafe_allow_html=True)

heart_disease_model = pickle.load(open('model/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('model/parkinsons_model.sav', 'rb'))
breast_cancer_model = keras.models.load_model('model/breast_cancer.keras')
diabetes_model = pickle.load(open('model/diabetes_model.sav', 'rb'))
diagnosis = pickle.load(open('model/svc.sav', 'rb'))

# Load CSVs
description = pd.read_csv("dataset/description.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
medications = pd.read_csv("dataset/medications.csv")
diets = pd.read_csv("dataset/diets.csv")
workout = pd.read_csv("dataset/workout_df.csv")

# Paste your symptoms_dict and diseases_list here
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    unknown = []
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(f"Unknown symptoms: {', '.join(unknown)}")
    # Uncomment the next line to debug input vector
    # print("Input vector:", input_vector)
    return diseases_list[diagnosis.predict([input_vector])[0]]



with st.sidebar:
    selected = option_menu(
        'Select disease',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction',
        'Diagnosis'],  # Added "Medical Resources"
        menu_icon='hospital-fill',
        icons=['droplet-half', 'heart', 'activity', 'gender-female', 'search-heart'],  # Added globe icon
        default_index=0
    )

    # Medical Disclaimer
    st.header("⚠️ Medical Disclaimer")
    st.markdown("""
    <div class="warning-box">
    <p><strong>Important:</strong> This tool is for research and educational purposes only. 
    It should not be used for actual medical diagnosis. Always consult with qualified 
    healthcare professionals for medical decisions.</p>
    </div>
    <style>
    .warning-box {
        background-color: #2e260c;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g. 2')
    with col2:
        Glucose = st.text_input('Glucose Level', placeholder='e.g. 120')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', placeholder='e.g. 80')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', placeholder='e.g. 20')
    with col2:
        Insulin = st.text_input('Insulin Level', placeholder='e.g. 85')
    with col3:
        BMI = st.text_input('BMI value', placeholder='e.g. 26.5')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder='e.g. 0.5')
    with col2:
        Age = st.text_input('Age of the Person', placeholder='e.g. 45')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                    BMI, DiabetesPedigreeFunction, Age]
        if any((x is None) or (str(x).strip() == "") for x in user_input):
            st.warning("Please fill all the columns before submitting.")
        else:
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                diab_diagnosis = '⚠️ The person is <b>diabetic</b>'
            else:
                diab_diagnosis = '✅ The person is <b>not diabetic</b>'

    if diab_diagnosis:
        st.markdown(f'<div class="success-box">{diab_diagnosis}</div>', unsafe_allow_html=True)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', placeholder='e.g. 54')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)', placeholder='e.g. 1')
    with col3:
        cp = st.text_input('Chest Pain Type (0–3)', placeholder='e.g. 2')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', placeholder='e.g. 130')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', placeholder='e.g. 250')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', placeholder='e.g. 0')
    with col1:
        restecg = st.text_input('Resting ECG Result (0–2)', placeholder='e.g. 1')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved', placeholder='e.g. 160')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)', placeholder='e.g. 0')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise', placeholder='e.g. 1.2')
    with col2:
        slope = st.text_input('Slope of the Peak ST Segment (0–2)', placeholder='e.g. 1')
    with col3:
        ca = st.text_input('Major Vessels Colored by Flourosopy (0–3)', placeholder='e.g. 0')
    with col1:
        thal = st.text_input('Thal (0=normal, 1=fixed defect, 2=reversible defect)', placeholder='e.g. 2')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]
        if any((x is None) or (str(x).strip() == "") for x in user_input):
            st.warning("Please fill all the columns before submitting.")
        else:
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                heart_diagnosis = '❤️ The person <b>has heart disease</b>'
            else:
                heart_diagnosis = '💚 The person <b>does not have any heart disease</b>'

    if heart_diagnosis:
        st.markdown(f'<div class="success-box">{heart_diagnosis}</div>', unsafe_allow_html=True)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    inputs = {}
    parkinsons_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    cols = st.columns(5)
    for i, feature in enumerate(parkinsons_features):
        with cols[i % 5]:
            inputs[feature] = st.text_input(feature, placeholder='e.g. 119.992')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        user_input = [inputs[f] for f in parkinsons_features]
        if any((x is None) or (str(x).strip() == "") for x in user_input):
            st.warning("Please fill all the columns before submitting.")
        else:
            user_input = [float(x) for x in user_input]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "🧠 The person <b>has Parkinson's disease</b>"
            else:
                parkinsons_diagnosis = "🧠 The person <b>does not have Parkinson's disease</b>"

    if parkinsons_diagnosis:
        st.markdown(f'<div class="success-box">{parkinsons_diagnosis}</div>', unsafe_allow_html=True)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title('👩‍⚕️ Breast Cancer Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.text_input('Radius Mean', placeholder='e.g. 17.99')
    with col2:
        texture_mean = st.text_input('Texture Mean', placeholder='e.g. 10.38')
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean', placeholder='e.g. 122.8')
    with col1:
        area_mean = st.text_input('Area Mean', placeholder='e.g. 1001')
    with col2:
        smoothness_mean = st.text_input('Smoothness Mean', placeholder='e.g. 0.1184')
    with col3:
        compactness_mean = st.text_input('Compactness Mean', placeholder='e.g. 0.2776')
    with col1:
        concavity_mean = st.text_input('Concavity Mean', placeholder='e.g. 0.3001')
    with col2:
        concave_points_mean = st.text_input('Concave Points Mean', placeholder='e.g. 0.1471')
    with col3:
        symmetry_mean = st.text_input('Symmetry Mean', placeholder='e.g. 0.2419')
    with col1:
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean', placeholder='e.g. 0.07871')

    breast_cancer_diagnosis = ''
    if st.button('Breast Cancer Test Result'):
        user_input = [radius_mean, texture_mean, perimeter_mean, area_mean,
                    smoothness_mean, compactness_mean, concavity_mean,
                    concave_points_mean, symmetry_mean, fractal_dimension_mean]
        if any((x is None) or (str(x).strip() == "") for x in user_input):
            st.warning("Please fill all the columns before submitting.")
        else:
            user_input = [float(x) for x in user_input]
            breast_cancer_prediction = breast_cancer_model.predict([user_input])
            if breast_cancer_prediction[0] == 1:
                breast_cancer_diagnosis = '🩺 The person <b>has Breast Cancer</b>'
            else:
                breast_cancer_diagnosis = '🩺 The person <b>does not have Breast Cancer</b>'

    if breast_cancer_diagnosis:
        st.markdown(f'<div class="success-box">{breast_cancer_diagnosis}</div>', unsafe_allow_html=True)


if selected == 'Diagnosis':
    st.title('🩺 Health Care Diagnosis')
    st.markdown("Enter your symptoms separated by commas (e.g. <i>itching, skin_rash, headache</i>):", unsafe_allow_html=True)
    symptoms_input = st.text_input('Symptoms', placeholder='e.g. itching, skin_rash, headache ,continuous_sneezing ,dehydration')

    # Initialize session state for diagnosis result
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = None

    if st.button('Recommendation'):
        if not symptoms_input.strip():
            st.warning("Please enter at least one symptom.")
        else:
            user_symptoms = [s.strip() for s in symptoms_input.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms if symptom.strip()]
            unknown = [s for s in user_symptoms if s not in symptoms_dict]
            if unknown:
                st.error(f"Unknown symptoms: {', '.join(unknown)}. Please check spelling.")
                st.session_state.diagnosis_result = None
            else:
                try:
                    predicted_disease = get_predicted_value(user_symptoms)
                    dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
                    st.session_state.diagnosis_result = {
                        "predicted_disease": predicted_disease,
                        "dis_des": dis_des,
                        "precautions": precautions,
                        "medications": medications,
                        "rec_diet": rec_diet,
                        "workout": workout
                    }
                except Exception as e:
                    st.error(f"❌ Could not process your symptoms. {e}")
                    st.session_state.diagnosis_result = None

    # Always show the radio and info blocks if result exists
    if st.session_state.diagnosis_result:
        st.markdown("### 📋 View Detailed Information")
        block = st.radio(
            "Select information to view:",
            ["🦠 Disease",
            "📖 Description",
            "🛡️ Precaution",
            "💊 Medication",
            "🏃 Workout",
            "🥗 Diets"],
            horizontal=True
        )
        res = st.session_state.diagnosis_result
        if block == "🦠 Disease":
            st.markdown(f"<div class='success-box'><b>Disease Prediction:</b> {res['predicted_disease']}</div>", unsafe_allow_html=True)
        elif block == "📖 Description":
            st.markdown(f"<div class='success-box'><b>Description:</b> {res['dis_des']}</div>", unsafe_allow_html=True)
        elif block == "🛡️ Precaution":
            st.markdown("<div class='success-box'><b>Precautions:</b><ul>" + "".join([f"<li>{p}</li>" for p in res['precautions'][0]]) + "</ul></div>", unsafe_allow_html=True)
        elif block == "💊 Medication":
            st.markdown("<div class='success-box'><b>Medications:</b><ul>" + "".join([f"<li>{m}</li>" for m in res['medications']]) + "</ul></div>", unsafe_allow_html=True)
        elif block == "🏃 Workout":
            if res['workout'] is not None and len(res['workout']) > 0:
                st.markdown("<div class='success-box'><b>Workout:</b><ul>" + "".join([f"<li>{w}</li>" for w in res['workout']]) + "</ul></div>", unsafe_allow_html=True)
            else:
                st.info("No workout recommendations available.")
        elif block == "🥗 Diets":
            if res['rec_diet']:
                st.markdown("<div class='success-box'><b>Recommended Diet:</b><ul>" + "".join([f"<li>{d}</li>" for d in res['rec_diet']]) + "</ul></div>", unsafe_allow_html=True)
            else:
                st.info("No diet recommendations available.")


if selected == 'QuickAid AI':
    st.markdown("""
    <style>
    @keyframes pulse {
      0% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.05); opacity: 0.8; }
      100% { transform: scale(1); opacity: 1; }
    }
    .pulse-text {
      animation: pulse 2s infinite;
      font-weight: bold;
      color: #16a34a;
    }
    .card:hover {
      box-shadow: 0 0 15px rgba(22, 163, 74, 0.4);
      transition: box-shadow 0.3s ease-in-out;
    }
    </style>

    <a href="https://ashiskumarmishra-ai-chat.hf.space/" target="_blank" style="text-decoration: none;">
        <div class="card" style="background-color: #f0fdf4; border: 2px solid #16a34a; border-radius: 12px; padding: 25px; text-align: center; margin: 20px 0; transition: 0.3s;">
            <h2 style="color: #166534; margin-bottom: 10px; font-size: 24px;">🤖 Your AI Health Assistant</h2>
            <p style="color: #14532d; font-size: 16px; margin: 0 auto; max-width: 500px;">
                Health made easy. Just speak and upload a photo describing your issue — and get instant, AI-generated remedies and insights tailored for you.
            </p>
            <button style="background-color: #16a34a; color: white; padding: 12px 24px; 
                           border-radius: 8px; border: none; font-weight: bold; font-size: 16px;
                           cursor: pointer; margin-top: 20px;">
                🌟 Launch AI Health Assistant
            </button>
            <p class="pulse-text" style="margin-top: 15px;">⬅ Click here to try it now!</p>
        </div>
    </a>
    """, unsafe_allow_html=True)


