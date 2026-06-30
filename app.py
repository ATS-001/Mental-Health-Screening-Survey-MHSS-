import streamlit as st
import re
import pickle

# Set page configuration
st.set_page_config(
    page_title="Mental Health Screening Survey (MHSS)",
    page_icon="📋",
    layout="centered"
)

# 1. Text Cleaning Function (Matches model.ipynb exactly)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. Load the Vectorizer and Model
@st.cache_resource
def load_model():
    try:
        with open('tfidf_model.pkl', 'rb') as f:
            vectorizer, model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("⚠️ 'tfidf_model.pkl' not found. Please ensure the model file is in this directory.")
        return None, None

vectorizer, model = load_model()

# 3. Streamlit UI Layout
st.title("📋 Mental Health Screening Survey (MHSS)")
st.write(
    "Please fill out this brief wellness questionnaire. Your responses will be analyzed "
    "by our backend AI model to evaluate potential indicators of emotional distress."
)

st.markdown("---")
# --- SYSTEM CONFIGURATION & CREDITS REGISTRY (From image_464e7c.png) ---

# Create two side-by-side columns
col1, col2 = st.columns(2)

with col1:
    with st.expander("ℹ️ System Configuration", expanded=False):
        st.markdown("""
        **Architecture:** Logistic Regression (Classification Model)
        
        **Feature Extraction:** TF-IDF Vectorizer
        
        **Execution Environment:** Local Vector Space (`tfidf_model.pkl`)
        """)

with col2:
    with st.expander("🛠️ Credits Registry", expanded=False):
        st.markdown("""
        **Creator / Engineer:** Aaron Thalakkottor Sooraj
        """)
        
        # Folder link mimicking the repo button
        st.button("📁 MHSS Repo", disabled=False)
        
        st.markdown("""
        **Project Baseline:** Day 4 of Projectathon conducted by µLearn LBSITW, AI x DS (29th June 2026)
        
        **Presented By:** Aiswarya Jayaprakash , Data Science IG LEAD, µLearn LBSITW
        """)

st.markdown("---")

# --- SURVEY QUESTIONS ---
st.subheader("Section 1: General Outlook & Mood")

q1 = st.radio(
    "1. How would you describe your overall perspective on your life right now?",
    options=[
        "I feel normal, bored sometimes, but generally okay about things.",
        "I feel like things are kinda tough, but I am getting through it.",
        "I feel like I have nothing to look forward to and cannot find a reason to go on."
    ],
    index=None
)

q2 = st.radio(
    "2. Which of the following best describes your recent daily experiences?",
    options=[
        "Enjoying simple things like a good movie, hanging out with guys/friends, or school/work.",
        "Feeling a bit tired or unmotivated lately.",
        "Dealing with severe depression, feeling completely empty, or wanting to end it all."
    ],
    index=None
)

q3 = st.radio(
    "3. Over the past two weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
    options=[
        "Not at all.",
        "Several days out of the week.",
        "Nearly every single day, completely overtaking my mood."
    ],
    index=None
)

q4 = st.radio(
    "4. How would you describe your current daily energy levels?",
    options=[
        "Normal and stable; I have the energy to complete my tasks.",
        "Slightly fatigued, sluggish, or dynamic depending on stress levels.",
        "Extremely exhausted, feeling heavy and unable to get out of bed most days."
    ],
    index=None
)

st.subheader("Section 2: Anxiety & Coping Mechanisms")

q5 = st.radio(
    "5. How often do you experience uncontrollable worrying, nervousness, or feeling on edge?",
    options=[
        "Rarely or not at all.",
        "Occasionally when facing specific deadlines or stressful moments.",
        "Constantly, making it extremely difficult to calm my mind down."
    ],
    index=None
)

q6 = st.radio(
    "6. When things get difficult or overwhelming, how do you typically respond?",
    options=[
        "I talk to friends, family, or take healthy breaks to clear my head.",
        "I try to ignore it or push through, though I feel highly stressed.",
        "I completely isolate myself, shut everyone out, and struggle to cope."
    ],
    index=None
)

st.subheader("Section 3: Sleep, Focus & Social Behavior")

q7 = st.radio(
    "7. How has your sleep pattern been over the past few weeks?",
    options=[
        "Good, restful, and regular sleeping hours.",
        "Irregular, either waking up in the middle of the night or sleeping too much.",
        "Severe insomnia, tracking completely broken sleep patterns or constant nightmares."
    ],
    index=None
)

q8 = st.radio(
    "8. Have you noticed changes in your ability to concentrate or make daily decisions?",
    options=[
        "No, my focus and decision-making are as sharp as usual.",
        "A little bit, I find myself drifting off or procrastinating more than normal.",
        "Yes, severe brain fog making it nearly impossible to focus on basic tasks."
    ],
    index=None
)

q9 = st.radio(
    "9. How connected do you feel to your friends, family, and peers lately?",
    options=[
        "Very connected; I actively interact with my social circles.",
        "Slightly distant; keeping to myself a bit more due to being busy or tired.",
        "Completely detached and lonely, feeling like an outsider to everyone."
    ],
    index=None
)

q10 = st.radio(
    "10. Which statement best matches your thoughts regarding your future?",
    options=[
        "I feel hopeful, motivated, or generally certain about where I'm heading.",
        "I feel uncertain, anxious, or a bit lost about the future.",
        "I feel absolute despair, dark thoughts, or like the future holds no value for me."
    ],
    index=None
)

st.subheader("Section 4: Additional Context (Optional)")
user_notes = st.text_area(
    "Is there anything specific on your mind, or any feelings you'd like to express in your own words?",
    placeholder="Feel free to type how you've been feeling lately...",
    height=100
)

st.markdown("---")

# --- ASSESSMENT LOGIC ---
if st.button("Submit Assessment", type="primary"):
    if vectorizer is not None and model is not None:
        
        # Combine all 10 survey selections and custom text into one large narrative block
        # This converts a structured survey into a text block our model understands
        combined_text = f"{q1} {q2} {q3} {q4} {q5} {q6} {q7} {q8} {q9} {q10} {user_notes}"
        
        # Preprocess the combined narrative text
        cleaned_input = clean_text(combined_text)
        
        # Vectorize and Predict
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        
        st.subheader("Screening Analysis Results")
        
        # Display results based on prediction
        if prediction == 1:
            st.error(f"🚨 **Screening Alert: Potential Indicators of Distress Detected**")
            st.write(f"The model detected risk patterns with a confidence level of **{probabilities[1] * 100:.2f}%**.")
            
            st.info(
                "💡 **Support is Available:** If you are experiencing overwhelming feelings, please know "
                "that you don't have to carry them alone. Consider speaking to a mental health professional, "
                "a trusted loved one, or contacting a local support helpline."
            )
        else:
            st.success(f"✅ **Screening Clear: No Immediate Distress Patterns Detected**")
            st.write(f"The model classified your responses as normal with a confidence level of **{probabilities[0] * 100:.2f}%**.")
            st.balloons()
            
# --- FOOTER SECTION ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888888; font-size: 0.85rem;">
        Developed by Aaron Thalakkottor Sooraj | Designed & Developed by ATS_PDZ | © 2026
    </div>
    """, 
    unsafe_allow_html=True
)