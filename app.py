import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from tensorflow import keras

try:
    # Chatbot imports
    from src.helper import download_hugging_face_embeddings
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    from langchain_groq import ChatGroq
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from dotenv import load_dotenv
    from src.prompt import system_prompt
    import pinecone
    CHATBOT_AVAILABLE = True
except ImportError as e:
    st.warning(f"Chatbot dependencies not available: {e}")
    CHATBOT_AVAILABLE = False


# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")



st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style='color: #4CAF50; margin: 0;'>Multiple Disease Prediction System</h1>
</div>
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
            /* Add cursor pointer to all interactive elements */
    div[data-baseweb="select"],
    button,
    .stSelectbox,
    .stButton > button,
    div[role="button"],
    div[role="listbox"],
    .stRadio label,
    input[type="radio"] + label,
    input[type="checkbox"] + label,
    a,
    .stTabs [data-baseweb="tab-list"] [role="tab"],
    .stWidgetLabel:has(+ div[data-baseweb="select"]) {
        cursor: pointer !important;
    }
    
    /* Make dropdown options show pointer cursor too */
    div[role="listbox"] div[role="option"] {
        cursor: pointer !important;
    }
    
    /* Ensure the dropdown arrow also shows pointer */
    div[data-baseweb="select"] svg {
        cursor: pointer !important;
    }
    
    /* For radio buttons and checkboxes */
    .stRadio > div > div > label,
    .stCheckbox > div > div > label {
        cursor: pointer !important;
    }
            
            
    /* More aggressive targeting for sidebar top space */
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }
    
    /* Target the option menu container */
    .css-1d391kg, .css-1544g2n {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Target all divs inside sidebar to remove top spacing */
    section[data-testid="stSidebar"] div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Fix menu top spacing specifically */
    .streamlit-option-menu {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Target main title in sidebar if present */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Force sidebar container to have no top padding */
    section[data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {
        padding-top: 0 !important;
    }  
    /* Add small top margin to the entire sidebar */
section[data-testid="stSidebar"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Apply top margin to direct children inside sidebar */
section[data-testid="stSidebar"] > div:first-child {
    margin-top: 30px !important;
    padding-top: 30px !important;
}

/* Extra fallback to push down menu */
div[data-testid="stSidebarNav"] {
    margin-top: 20px !important;
    padding-top: 20px !important;
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


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def initialize_chatbot():
    if not CHATBOT_AVAILABLE:
        return None
    try:
        # Simple test first
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found")
            
        # Initialize just the LLM for testing (without Pinecone)
        llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
        
        # Test the LLM connection
        test_response = llm.invoke("Hello")
        
        # Return the LLM for simple chat
        return llm
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None


with st.sidebar:
    selected = option_menu(
        'Health Services',
        ['About','Disease Prediction', 'Diagnosis', 'Medical Chatbot', 
         'Instant Medication', 'Prescription Analyser'],  
        menu_icon='hospital-fill',
        icons=['activity', 'search-heart', 'chat-dots', 'capsule', 'file-medical'],  
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
    



# Disease Prediction Section (with dropdown)
if selected == 'Disease Prediction':
    st.title('👇 Select Disease Type to Predict')
    
    # Dropdown for disease selection
    disease_type = st.selectbox(
        "",
        ["Diabetes", "Heart Disease", "Parkinson's Disease", "Breast Cancer"],
        index=0,
        format_func=lambda x: f"🔍 {x} Prediction"
    )
    
    # Diabetes Prediction
    if disease_type == "Diabetes":
        st.subheader('🩺 Diabetes Prediction')

        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g. 2', key="diabetes_1")
        with col2:
            Glucose = st.text_input('Glucose Level', placeholder='e.g. 120', key="diabetes_2")
        with col3:
            BloodPressure = st.text_input('Blood Pressure value', placeholder='e.g. 80', key="diabetes_3")
        with col1:
            SkinThickness = st.text_input('Skin Thickness value', placeholder='e.g. 20', key="diabetes_4")
        with col2:
            Insulin = st.text_input('Insulin Level', placeholder='e.g. 85', key="diabetes_5")
        with col3:
            BMI = st.text_input('BMI value', placeholder='e.g. 26.5', key="diabetes_6")
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder='e.g. 0.5', key="diabetes_7")
        with col2:
            Age = st.text_input('Age of the Person', placeholder='e.g. 45', key="diabetes_8")

        diab_diagnosis = ''
        if st.button('Diabetes Test Result', key="diabetes_btn"):
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

    # Heart Disease Prediction
    elif disease_type == "Heart Disease":
        st.subheader('❤️ Heart Disease Prediction')

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input('Age', placeholder='e.g. 54', key="heart_1")
        with col2:
            sex = st.text_input('Sex (1=Male, 0=Female)', placeholder='e.g. 1', key="heart_2")
        with col3:
            cp = st.text_input('Chest Pain Type (0–3)', placeholder='e.g. 2', key="heart_3")
        with col1:
            trestbps = st.text_input('Resting Blood Pressure', placeholder='e.g. 130', key="heart_4")
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl', placeholder='e.g. 250', key="heart_5")
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', placeholder='e.g. 0', key="heart_6")
        with col1:
            restecg = st.text_input('Resting ECG Result (0–2)', placeholder='e.g. 1', key="heart_7")
        with col2:
            thalach = st.text_input('Maximum Heart Rate Achieved', placeholder='e.g. 160', key="heart_8")
        with col3:
            exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)', placeholder='e.g. 0', key="heart_9")
        with col1:
            oldpeak = st.text_input('ST Depression Induced by Exercise', placeholder='e.g. 1.2', key="heart_10")
        with col2:
            slope = st.text_input('Slope of the Peak ST Segment (0–2)', placeholder='e.g. 1', key="heart_11")
        with col3:
            ca = st.text_input('Major Vessels Colored by Flourosopy (0–3)', placeholder='e.g. 0', key="heart_12")
        with col1:
            thal = st.text_input('Thal (0=normal, 1=fixed defect, 2=reversible defect)', placeholder='e.g. 2', key="heart_13")

        heart_diagnosis = ''
        if st.button('Heart Disease Test Result', key="heart_btn"):
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

    # Parkinson's Disease Prediction
    elif disease_type == "Parkinson's Disease":
        st.subheader('🧠 Parkinson\'s Disease Prediction')
        
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
                inputs[feature] = st.text_input(feature, placeholder='e.g. 119.992', key=f"park_{i}")

        parkinsons_diagnosis = ''
        if st.button("Parkinson's Test Result", key="park_btn"):
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
    
    # Breast Cancer Prediction
    elif disease_type == "Breast Cancer":
        st.subheader('👩‍⚕️ Breast Cancer Prediction')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            radius_mean = st.text_input('Radius Mean', placeholder='e.g. 17.99', key="bc_1")
        with col2:
            texture_mean = st.text_input('Texture Mean', placeholder='e.g. 10.38', key="bc_2")
        with col3:
            perimeter_mean = st.text_input('Perimeter Mean', placeholder='e.g. 122.8', key="bc_3")
        with col1:
            area_mean = st.text_input('Area Mean', placeholder='e.g. 1001', key="bc_4")
        with col2:
            smoothness_mean = st.text_input('Smoothness Mean', placeholder='e.g. 0.1184', key="bc_5")
        with col3:
            compactness_mean = st.text_input('Compactness Mean', placeholder='e.g. 0.2776', key="bc_6")
        with col1:
            concavity_mean = st.text_input('Concavity Mean', placeholder='e.g. 0.3001', key="bc_7")
        with col2:
            concave_points_mean = st.text_input('Concave Points Mean', placeholder='e.g. 0.1471', key="bc_8")
        with col3:
            symmetry_mean = st.text_input('Symmetry Mean', placeholder='e.g. 0.2419', key="bc_9")
        with col1:
            fractal_dimension_mean = st.text_input('Fractal Dimension Mean', placeholder='e.g. 0.07871', key="bc_10")

        breast_cancer_diagnosis = ''
        if st.button('Breast Cancer Test Result', key="bc_btn"):
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

    # Add styling for the disease selection dropdown
    st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        background-color: #4CAF50;
        color: white !important;
        border-radius: 10px;
        padding: 5px;
        font-weight: bold;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #45a049;
    }
    div[data-baseweb="select"] svg {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)



# Medical Diagnosis
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



# Medical Chatbot Page
if selected == 'Medical Chatbot':


    st.title('🤖 Medical Chatbot')
    st.markdown("Ask any medical question and get AI-powered responses.")
    
    if not CHATBOT_AVAILABLE:
        st.error("❌ Chatbot dependencies not available. Please install required packages.")
        st.code("pip install langchain-groq python-dotenv", language="bash")
        st.stop()
    
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()
    
    # Initialize simple LLM
    try:
        llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {e}")
        st.stop()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    user_input = st.text_input(
        "Type your medical question:", 
        key="chatbot_input",
        placeholder="e.g. What are the symptoms of diabetes?"
    )
    
    # Process user input
    if st.button("Send 📤", key="send_chat") and user_input:
        if user_input.strip():
            with st.spinner("🔍 Processing your question..."):
                try:
                    # Create medical prompt
                    prompt = f"""You are a helpful medical AI assistant. Provide accurate medical information for: {user_input}
                    
Always remind users to consult healthcare professionals for medical decisions."""
                    
                    # Get response
                    response = llm.invoke(prompt)
                    bot_response = response.content
                    
                    # Save interaction
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("bot", bot_response))
                    
                    # Refresh page
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        chat_pairs = list(zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]))
        
        for user, bot in reversed(chat_pairs):
            user_msg = user[1]
            bot_msg = bot[1]
            
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <div style="background-color: #58cc71; padding: 12px; border-radius: 15px 0px 15px 15px; 
                                max-width: 80%; color: white; word-wrap: break-word;">
                        <strong>👤 You:</strong><br>{user_msg}
                    </div>
                </div>
                <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
                    <div style="background-color: #52acff; padding: 12px; border-radius: 0px 15px 15px 15px; 
                                max-width: 80%; color: white; word-wrap: break-word;">
                        <strong>🤖 Medical AI:</strong><br>{bot_msg}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.markdown("""
        <div class="success-box">
            <h3>🩺 Welcome to Medical AI Assistant!</h3>
            <p>I can help you with:</p>
            <ul>
                <li>🔍 Medical symptoms and conditions</li>
                <li>💊 Medication information</li>
                <li>🏥 Treatment options</li>
            </ul>
            <p><strong>Start by asking a question above!</strong></p>
        </div>
        """, unsafe_allow_html=True)



# Instant medication
if selected == 'Instant Medication':
    st.title('💊 Instant Medication Recommendations')
    st.markdown("Get quick medication suggestions based on your symptoms")
    
    # Quick link to external AI service
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                border: 2px solid #16a34a; border-radius: 15px; 
                padding: 25px; margin: 20px 0; text-align: center;">
        <h3 style="color: #16a34a; margin-bottom: 15px;">🤖 AI-Powered Medication Assistant</h3>
        <p style="color: #166534; margin-bottom: 20px;">
            Get instant, personalized medication recommendations using our advanced AI system
        </p>
        <a href="https://ashiskumarmishra-ai-chat.hf.space/" target="_blank" 
           style="background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
                  color: white; padding: 15px 30px; border-radius: 12px;
                  text-decoration: none; font-weight: bold; font-size: 16px;
                  display: inline-block; transition: all 0.3s ease;
                  box-shadow: 0 4px 15px rgba(22, 163, 74, 0.3);">
            🚀 Launch AI Medication Assistant
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📖 How it works:")
    st.markdown("""
    1. **📤 Upload** your symptoms or prescription image  
    2. **🎙️ Speak** your health issue or query using voice input  
    3. **🤖 AI Processing** understands your symptoms in real time  
    4. **📄 Receive** instant medication suggestions both as **text** and **voice**  
    5. **🚀 Launch the AI Assistant** for more detailed health support
    """)



# Prescription Analyser Page  
if selected == 'Prescription Analyser':
    st.title('📋 Prescription Analyser')
    st.markdown("Upload and analyze your medical prescription with AI assistance")
    
    # Quick link to external AI service
    st.markdown("""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                border: 2px solid #3b82f6; border-radius: 15px; 
                padding: 25px; margin: 20px 0; text-align: center;">
        <h3 style="color: #1e40af; margin-bottom: 15px;">🤖 AI Prescription Analysis</h3>
        <p style="color: #1e3a8a; margin-bottom: 20px;">
            Upload your prescription image and get instant AI analysis with our advanced system
        </p>
        <a href="https://prescriptionashis.streamlit.app/" target="_blank" 
           style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                  color: white; padding: 15px 30px; border-radius: 12px;
                  text-decoration: none; font-weight: bold; font-size: 16px;
                  display: inline-block; transition: all 0.3s ease;
                  box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">
            🔍 Launch AI Prescription Analyzer
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📖 How it works:")
    st.markdown("""
    1. **📄 Upload** your medical prescription (PDF or image)  
    2. **🔍 AI reads** the content and extracts drug names, dosages, and instructions  
    3. **💬 Ask anything** related to the uploaded prescription (e.g., side effects, dosage timing)  
    4. **📚 Get responses** directly from your document using our AI system  
    5. **🚀 Launch the Analyzer** for an interactive Q&A experience
    """)



# Add this at the end of your file, after all your other sections

if selected == 'About':
    # Add custom styling for better header visibility
    st.markdown("""
    <style>
    .feature-header {
        color: #4CAF50;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .section-box {
        background-color: #4CAF50;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .navigation-item {
        background-color: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    .navigation-item strong {
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title('ℹ️ About Health Assistant')
    
    # Introduction with clearer description
    st.markdown("""
    <div class="section-box">
        <p>This application provides AI-powered health diagnostics and medical information using machine learning models. It's designed to help you assess health conditions, learn about diseases, and access medication information - all in one place.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # What's Included with navigation information
    st.subheader('🛠️ Features & Where to Find Them')
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-header">Disease Prediction</div>
        <div class="navigation-item">
            <strong>Where to find:</strong> Select "Disease Prediction" in the sidebar menu
        </div>
        <ul>
            <li><strong>Diabetes Prediction</strong>: Enter glucose levels, BMI and other clinical data</li>
            <li><strong>Heart Disease Prediction</strong>: Input cardiovascular parameters</li>
            <li><strong>Parkinson's Disease Detection</strong>: Provide voice analysis measurements</li>
            <li><strong>Breast Cancer Prediction</strong>: Enter tumor characteristic values</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-header">Health Services</div>
        <div class="navigation-item">
            <strong>Where to find:</strong> Select each service individually in the sidebar
        </div>
        <ul>
            <li><strong>Medical Diagnosis</strong>: Enter symptoms in the "Diagnosis" section</li>
            <li><strong>Medical Chatbot</strong>: Ask health questions in the "Medical Chatbot" section</li>
            <li><strong>Medication Recommendations</strong>: Find in "Instant Medication" section</li>
            <li><strong>Prescription Analysis</strong>: Access in "Prescription Analyser" section</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Navigation Guide
    st.subheader('🧭 Quick Navigation Guide')
    st.markdown("""
    <div class="section-box">
        <p>Here's how to navigate the application:</p>
        <ul>
            <li><strong>For disease risk assessment:</strong> Click "Disease Prediction" in the sidebar and select the disease type</li>
            <li><strong>For symptom-based diagnosis:</strong> Select "Diagnosis" and enter your symptoms separated by commas</li>
            <li><strong>To ask medical questions:</strong> Choose "Medical Chatbot" and type your health queries</li>
            <li><strong>For medication assistance:</strong> Use "Instant Medication" and follow the link to the AI assistant</li>
            <li><strong>To analyze prescriptions:</strong> Go to "Prescription Analyser" and upload your prescription</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Technologies Used
    st.subheader('🧪 Technologies Used')
    st.markdown("""
    - **Machine Learning**: scikit-learn, TensorFlow
    - **Web Interface**: Streamlit
    - **Natural Language Processing**: LLMs for chatbot functionality
    - **Data Processing**: Pandas, NumPy
    """)
    
    # How It Works
    st.subheader('⚙️ How It Works')
    st.markdown("""
    This system combines multiple machine learning models trained on various medical datasets to provide accurate disease predictions:
    
    1. **Input Collection**: You provide relevant health parameters or symptoms
    2. **Data Processing**: System normalizes and prepares your inputs
    3. **Model Prediction**: Trained ML models analyze the data
    4. **Results Generation**: You receive prediction results with recommendations
    5. **Follow-up Support**: Where appropriate, additional resources are provided
    """)
    
    # Development Team
    st.subheader('👨‍💻 Development')
    st.markdown("""
    Developed as part of a machine learning project demonstrating healthcare applications of artificial intelligence.
    
    **Version:** 1.0.0
    """)
    
    # Footer with contact
    st.markdown("""
    <div style="background-color: #4CAF50; padding: 15px; border-radius: 5px; margin-top: 30px; text-align: center; color: white;">
        <h4 style="margin:0;">Need help? Have suggestions?</h4>
        <p>Contact us at <a href="mailto:mishralucky074@gmail.com" style="color: white; text-decoration: underline;">mishralucky074@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

        