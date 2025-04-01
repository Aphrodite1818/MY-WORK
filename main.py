import pickle as pkl
import streamlit as st
import pandas as pd
from PIL import Image

# Set page config
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Display header with image
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://drjayanam.com/wp-content/uploads/2023/01/infografic-design-03.webp", 
             width=200)
with col2:
    st.title("Breast Cancer Risk Assessment")
    st.write("This tool helps assess your risk of breast cancer based on clinical factors.")

# Cache the model loading for better performance
@st.cache_resource
def load_model():
    try:
        with open('Deployment_model.pkl', 'rb') as model_file:
            return pkl.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'Deployment_model.pkl' is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

bot = load_model()

# Encoding mappings (must match training)
ENCODINGS = {
    'Menopause': {'No': 0, 'Yes': 1},
    'Breast': {'Left': 0, 'None': 1, 'Right': 2},
    'Metastasis': {'No': 0, 'None': 1, 'Yes': 2},
    'BreastQuadrant': {
        'Lower inner': 0, 
        'Lower outer': 1, 
        'None': 2, 
        'Upper inner': 3, 
        'Upper outer': 4
    },
    'History': {'No': 0, 'Yes': 2, 'None': 1}
}

def get_user_inputs():
    """Collect and validate user inputs"""
    with st.form("patient_details"):
        st.header("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.slider("Year of examination", 2000, 2030, 2023)
            age = st.slider("Age", 18, 100, 40)
            menopause = st.selectbox("Menopause status", options=list(ENCODINGS['Menopause'].keys()))
            tumor_size = st.slider("Tumor size (cm)", 0, 20, 1)
            inv_nodes = st.selectbox("Number of involved lymph nodes", options=[0, 1, 2])
            
        with col2:
            breast = st.selectbox("Affected breast", options=list(ENCODINGS['Breast'].keys()))
            metastasis = st.selectbox("Metastasis present?", options=list(ENCODINGS['Metastasis'].keys()))
            breast_quadrant = st.selectbox("Breast quadrant", options=list(ENCODINGS['BreastQuadrant'].keys()))
            history = st.selectbox("Family history of breast cancer", options=list(ENCODINGS['History'].keys()))
        
        submitted = st.form_submit_button("Assess Risk")
        
        if submitted:
            # Convert categorical inputs to encoded values
            inputs = [
                year,
                age,
                ENCODINGS['Menopause'][menopause],
                tumor_size,
                inv_nodes,
                ENCODINGS['Breast'][breast],
                ENCODINGS['Metastasis'][metastasis],
                ENCODINGS['BreastQuadrant'][breast_quadrant],
                ENCODINGS['History'][history]
            ]
            return inputs
    return None

def make_prediction(input_data):
    """Make prediction using the trained model"""
    if bot is None:
        return "Error: Model not available", None
    
    try:
        # Create DataFrame with correct feature names
        feature_names = [
            'Year', 'Age', 'Menopause', 'Tumor Size (cm)', 'Inv-Nodes',
            'Breast', 'Metastasis', 'Breast Quadrant', 'History'
        ]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Get prediction and probability
        prediction = bot.predict(input_df)[0]
        proba = bot.predict_proba(input_df)[0]
        
        return prediction, proba
    except Exception as e:
        return f"Prediction error: {str(e)}", None

def display_results(prediction, probability):
    """Display prediction results with styling"""
    st.subheader("Assessment Results")
    
    if isinstance(prediction, str):  # Error case
        st.error(prediction)
    else:
        # Get prediction label
        result = "Benign" if prediction == 0 else "Malignant"
        confidence = probability[prediction] * 100
        
        # Display with appropriate color
        if prediction == 0:
            st.success(f"**Result:** {result} ({(confidence):.1f}% confidence)")
            st.balloons()
            st.write("""
            This suggests a low risk of breast cancer. However, regular screenings 
            are still recommended based on your age and risk factors.
            """)
        else:
            st.error(f"**Result:** {result} ({(confidence):.1f}% confidence)")
            st.write("""
            **Please consult with a healthcare professional immediately** 
            for further evaluation and diagnostic testing.
            """)
        
        # Show probability breakdown
        with st.expander("See detailed probabilities"):
            st.write(f"Benign probability: {(probability[0]*100):.1f}%")
            st.write(f"Malignant probability: {(probability[1]*100):.1f}%")

def main():
    """Main app function"""
    inputs = get_user_inputs()
    
    if inputs:
        # Show spinner while processing
        with st.spinner("Analyzing your information..."):
            prediction, probability = make_prediction(inputs)
            display_results(prediction, probability)
    
    # Add informational section
    st.markdown("---")
    st.subheader("About This Tool")
    st.write("""
    This predictive model was developed using clinical breast cancer data. 
    It assesses risk based on factors like tumor size, lymph node involvement, 
    and patient history.
    
    **Note:** This tool is for informational purposes only and should not 
    replace professional medical advice. Always consult with a healthcare 
    provider for diagnosis and treatment.
    """)

if __name__ == '__main__':
    main()