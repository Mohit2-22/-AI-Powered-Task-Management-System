import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime
import pandas as pd

# Initialize session state for storing tasks
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Developer assignments
DEVELOPER_MAPPING = {
    "Capella": "Rohit",
    "Platform": "Mohit",
    "Community": "Anjali"
}

# --- Load Models ---
try:
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    classifier_model = joblib.load('classifier_model.joblib')
    priority_model = joblib.load('priority_model.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please make sure the .joblib files are in the correct directory.")
    st.stop()

# --- Text Processing ---
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

# --- Prediction Function ---
def predict_task_attributes(text):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf.transform([processed_text])
    category = classifier_model.predict(vectorized_text)[0]
    priority = priority_model.predict(vectorized_text)[0]
    return category, priority





# --- Streamlit Web App ---
st.set_page_config(page_title="AI Task Manager", page_icon="🤖", layout="wide")

# --- Animated Center Title ---
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 59px; 
                   animation: pulse 2s infinite;
                   text-shadow: 2px 2px 10px rgba(229,9,20,0.6);
                   margin-bottom: 10px;'>
            🤖 AI-Powered Task Management System
        </h1>
       
    <style>
        @keyframes pulse {
            0% { color: #e50914; text-shadow: 2px 2px 10px rgba(229,9,20,0.6); }
            50% { color: #ff001a; text-shadow: 0 0 20px rgba(255,0,26,0.8); }
            100% { color: #e50914; text-shadow: 2px 2px 10px rgba(229,9,20,0.6); }
        }
    </style>
""", unsafe_allow_html=True)


st.write("Enter a task description below and watch the AI automatically classify and prioritize & Assign the task to Developer it.")


# --- Tabs ---
tab1, tab2 = st.tabs(["📝 New Task", "📋 View Tasks"])

with tab1:
    st.session_state.user_input = st.text_area(
        "Enter a new task description here:",
        value=st.session_state.user_input,
        height=140,
        placeholder="e.g., The login button is not working on the payment screen..."
    )

    col1, col2 = st.columns([1, 0.3])
    with col1:
        analyze_clicked = st.button("🔍 Analyze Task", type="primary")
    with col2:
        clear_clicked = st.button("🧹 Clear")

    # Clear button functionality
    if clear_clicked:
        st.session_state.user_input = ""
        st.stop()

    if analyze_clicked:
        user_input = st.session_state.user_input.strip()
        if user_input:
            with st.spinner('Analyzing...'):
                predicted_category, predicted_priority = predict_task_attributes(user_input)
                assigned_developer = DEVELOPER_MAPPING.get(predicted_category, "Unassigned")

                task = {
                    "description": user_input,
                    "category": predicted_category,
                    "priority": predicted_priority,
                    "developer": assigned_developer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                st.session_state.tasks.append(task)
                st.success("✅ Task analyzed and assigned successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="📂 Category", value=predicted_category)
                with col2:
                    st.metric(label="🎯 Priority", value=predicted_priority)
                with col3:
                    st.metric(label="👤 Assigned To", value=assigned_developer)
        else:
            st.warning("Please enter a task description to analyze.")

with tab2:
    st.subheader("📋 Assigned Tasks Log")

    if st.session_state.tasks:
        df = pd.DataFrame(st.session_state.tasks)
        for idx, task in enumerate(st.session_state.tasks):
            with st.expander(f"Task {idx + 1}: {task['description'][:50]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Description:** {task['description']}")
                    st.markdown(f"**Category:** {task['category']}")
                    st.markdown(f"**Priority:** {task['priority']}")
                with col2:
                    st.markdown(f"**Assigned to:** {task['developer']}")
                    st.markdown(f"**Timestamp:** {task['timestamp']}")

        st.download_button(
            label="📥 Download Tasks as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="tasks_export.csv",
            mime="text/csv"
        )
    else:
        st.info("No tasks have been assigned yet!")
