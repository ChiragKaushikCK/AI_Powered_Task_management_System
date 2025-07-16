import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.preprocessing import LabelEncoder 
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="AI-Powered Task Management",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a highly polished UI, adapted for Task Management
st.markdown("""
<style>
            @media (max-width: 768px) {
    .card { padding: 12px; min-height: unset; }
    .header h1 { font-size: 1.5em; }
    .header p { font-size: 1em; }
    }
    /* Global Styles & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    html, body, [class*="st-emotion-cache"] {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #333;
    }

    body {
    background: linear-gradient(120deg, #f8ffae 0%, #43cea2 100%);
    min-height: 100vh;
    overflow-x: hidden;
    
    /* Main App Container */
    .stApp > header { display: none; }
    .stApp { background: none; }
    .st-emotion-cache-z5fcl4, .st-emotion-cache-km4b0p, .st-emotion-cache-18ni7ap { 
        padding: 20px;
        max-width: 1600px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.08); /* Subtle transparent background */
        border-radius: 25px;
        box-shadow: 0 30px 60px rgba(0,0,0,0.3);
        backdrop-filter: blur(12px); /* Glassy effect */
        border: 1px solid rgba(255,255,255,0.15);
        animation: fadeIn 0.8s ease-out;
        min-height: calc(100vh - 40px);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Header Styling */
    .header {
        text-align: center;
        margin-bottom: 50px;
        color: white;
    }

    .header h1 {
        font-size: 3.8em;
        margin-bottom: 15px;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.6);
        letter-spacing: -1.5px;
        font-weight: 900;
    }

    .header p {
        font-size: 1.4em;
        opacity: 0.95;
        font-weight: 300;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Card Styling */

    }
    .card {
    background: rgba(255,255,255,0.85);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
    backdrop-filter: blur(12px);
    border: 1.5px solid rgba(255,255,255,0.25);
    transition: transform 0.25s, box-shadow 0.25s;
    min-height: 350px;
    }
    .card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 16px 40px 0 rgba(31,38,135,0.22);
    }
    .card h2 {
        color: #2c3e50; /* Darker header color */
        margin-bottom: 25px;
        font-size: 1.9em;
        display: flex;
        align-items: center;
        gap: 15px;
        font-weight: 700;
        border-bottom: 2px solid #ecf0f1; /* Light grey separator */
        padding-bottom: 15px;
    }

    .icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(45deg, #2980b9, #3498db); /* Blue gradient */
        border-radius: 50%;
        display: inline-flex; 
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.9em; 
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }

    /* Streamlit Input & Button General Overrides */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { 
        padding: 16px;
        border: 2px solid #bdc3c7; /* Lighter grey border */
        border-radius: 12px;
        font-size: 18px;
        width: 100%;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
        background-color: #f8f9fa; /* Very light background */
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        outline: none;
        color: #2c3e50; /* Darker text on focus */
        border-color: #3498db; /* Blue focus */
        box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.3);
    }
    /* Selectbox styling */
    div[data-testid="stSelectbox"] > div[role="listbox"] {
        background-color: #51A1FE;
        border: 2px solid #bdc3c7;
        border-radius: 12px;
    }
    div[data-testid="stSelectbox"] > div > div { /* For actual select box */
        padding: 12px;
        font-size: 18px;
    }
    div[data-testid="stSelectbox"] div[role="option"] {
        padding: 10px 15px;
        font-size: 16px;
    }


    .stButton > button { 
        padding: 16px 30px;
        background: linear-gradient(45deg, #2ecc71, #27ae60); /* Green gradient */
        color: white;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 700;
        font-size: 1.2em;
        transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
        box-shadow: 0 10px 20px rgba(46, 204, 113, 0.4);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(46, 204, 113, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
        opacity: 0.85;
    }

    /* Task Card Styling */
    .task-card-streamlit {
        background: linear-gradient(135deg, #fefefe, #f0f3f6);
        border-radius: 18px; 
        padding: 25px;
        transition: all 0.3s ease;
        cursor: default; 
        border: 2px solid #e2e8f0;
        display: flex;
        flex-direction: column;
        justify-content: space-between; 
        height: 100%; 
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    .task-card-streamlit:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 36px rgba(0,0,0,0.12);
        border-color: #3498db; /* Blue on hover */
    }

    .task-card-streamlit.status-completed {
        border-color: #2ecc71; /* Green for completed */
        background: linear-gradient(135deg, #e6ffe6, #c6f6d5); 
        box-shadow: 0 10px 20px rgba(46, 204, 113, 0.15); 
    }
    .task-card-streamlit.status-in-progress {
        border-color: #f1c40f; /* Yellow for in progress */
        background: linear-gradient(135deg, #fffbe6, #fff3b3);
        box-shadow: 0 10px 20px rgba(241, 196, 15, 0.15);
    }
    .task-card-streamlit.status-open {
        border-color: #e74c3c; /* Red for open */
        background: linear-gradient(135deg, #ffe6e6, #ffb3b3);
        box-shadow: 0 10px 20px rgba(231, 76, 60, 0.15);
    }


    .task-card-streamlit .task-title {
        font-weight: 800;
        font-size: 1.3em; 
        margin-bottom: 10px;
        color: #2d3748;
        line-height: 1.3;
    }

    .task-card-streamlit .task-details {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        font-size: 15px;
    }

    .task-card-streamlit .task-priority {
        font-weight: 600;
        color: #718096;
    }

    .task-card-streamlit .task-categories {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 15px;
        min-height: 30px; 
    }

    .task-card-streamlit .category-tag {
        background: rgba(52, 152, 219, 0.1); /* Blue for category */
        color: #3498db; 
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(52, 152, 219, 0.2);
    }
    /* Priority specific styles for tags */
    .priority-tag-High { background: rgba(231, 76, 60, 0.1); color: #e74c3c; border-color: rgba(231, 76, 60, 0.2); }
    .priority-tag-Medium { background: rgba(241, 196, 15, 0.1); color: #f1c40f; border-color: rgba(241, 196, 15, 0.2); }
    .priority-tag-Low { background: rgba(46, 204, 113, 0.1); color: #2ecc71; border-color: rgba(46, 204, 113, 0.2); }


    .task-card-streamlit .task-description {
        color: #718096;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 20px;
        flex-grow: 1; 
    }

    .task-card-streamlit .status-section {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-top: auto; 
    }
    .status-dropdown .stSelectbox > div > div[role="listbox"] {
        border-radius: 8px;
        border-color: #bdc3c7;
        padding: 5px;
        font-size: 14px;
    }
    .status-dropdown .stSelectbox > div > div > div {
        padding: 5px 10px;
        font-size: 14px;
    }

    /* Predicted Task Item Styling */
    .predicted-task-item-streamlit {
        background: linear-gradient(135deg, #f8fafc, #eef1f4);
        padding: 20px;
        border-radius: 15px;
        border-left: 7px solid #3498db; /* Blue border for predictions */
        transition: all 0.3s ease;
        cursor: default;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    .predicted-task-item-streamlit:hover {
        transform: translateX(10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        border-left-color: #2980b9; 
    }

    .predicted-task-item-streamlit .pred-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .predicted-task-item-streamlit .pred-title {
        font-weight: 700;
        color: #2d3748;
        font-size: 1.2em;
    }

    .prediction-confidence {
        background: rgba(46, 204, 113, 0.1); /* Green for confidence */
        color: #2ecc71; 
        padding: 7px 15px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 600;
        border: 1px solid rgba(46, 204, 113, 0.2);
    }

    .predicted-task-item-streamlit .pred-details {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        font-size: 14px;
        color: #718096;
    }

    .predicted-task-item-streamlit .pred-categories {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .predicted-task-item-streamlit .pred-category-tag { 
        background: rgba(52, 152, 219, 0.08); 
        color: #3498db;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
    }

    .predicted-priority-tag {
        background: rgba(231, 76, 60, 0.1); 
        color: #e74c3c; 
        padding: 7px 15px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 600;
        border: 1px solid rgba(231, 76, 60, 0.2);
    }

    /* Model Tabs / Navigation */
    div[data-testid="stForm"] > div > div > div > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 15px;
        margin-bottom: 25px;
    }
    .stRadio > label {
        font-weight: 600;
        color: #4a5568;
        font-size: 16px;
        padding: 12px 28px;
        border-radius: 30px;
        background: #e9eff5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid #d0dbe7;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-grow: 1;
    }
    .stRadio > label:hover {
        background: #dae5f2;
        transform: translateY(-2px);
    }
    .stRadio > label[data-baseweb="radio"] > div > div[data-testid="stDecoration"] {
        background: linear-gradient(45deg, #3498db, #2980b9) !important; /* Blue gradient for active tab */
        color: white !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 16px rgba(52, 152, 219, 0.3) !important;
        border-color: transparent !important; 
    }
    .stRadio > label > div > div[data-testid="stDecoration"] div { display: none !important; }
    .stRadio > label > div > div { justify-content: center; width: 100%; }

    /* Search & Filter (Task List) */
    .stTextInput input {
        padding: 16px;
        border: 2px solid #bdc3c7;
        border-radius: 12px;
        font-size: 18px;
        margin-bottom: 15px;
    }

    /* Genre filter buttons */
    .filter-buttons-container { /* Renamed for task management context */
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-bottom: 25px;
    }
    .filter-buttons-container .stButton > button {
        padding: 8px 18px;
        background: #e9eff5;
        color: #4a5568;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #d0dbe7;
        width: auto;
    }
    .filter-buttons-container .stButton > button:hover {
        background: #dae5f2;
        transform: translateY(-1px);
    }
    .filter-buttons-container .stButton > button[data-testid^="stButton-primary"] {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
        border-color: transparent !important;
    }


    /* Metrics Section */
    .metric-card-streamlit {
        background: linear-gradient(135deg, #ffffff, #f7fafc);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        height: 100%; 
    }

    .metric-card-streamlit .metric-value {
        font-size: 2.2em;
        font-weight: 800;
        color: #3498db; /* Blue for metrics */
        margin-bottom: 8px;
    }

    .metric-card-streamlit .metric-label {
        color: #718096;
        font-weight: 700;
        font-size: 13px; 
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .chart-container-streamlit {
        background: white;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        height: 380px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .empty-state-streamlit {
        text-align: center;
        padding: 50px 20px;
        color: #718096;
        flex-grow: 1; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 2px dashed #bdc3c7; /* Lighter dashed border */
        border-radius: 15px;
        margin: 10px;
        min-height: 250px;
    }

    .empty-state-streamlit .empty-icon {
        font-size: 4.5em;
        margin-bottom: 25px;
    }
    .empty-state-streamlit p {
        font-size: 1.2em;
        max-width: 350px;
        line-height: 1.4;
    }

    .tips-box-streamlit { /* Renamed from personalization-tips */
        background: rgba(52, 152, 219, 0.08); 
        border: 1px solid rgba(52, 152, 219, 0.15); 
        border-radius: 15px;
        padding: 20px;
        margin-top: 30px;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.08);
    }

    .tips-box-streamlit .tips-title {
        font-weight: 700;
        color: #3498db;
        margin-bottom: 12px;
        font-size: 1.15em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .tips-box-streamlit .tips-text {
        font-size: 15px;
        color: #4a5568;
        line-height: 1.6;
    }

    /* Streamlit columns padding adjustment */
    .st-emotion-cache-1jmve3k, .st-emotion-cache-s2g60g, .st-emotion-cache-h5gxhg { 
        gap: 30px; 
    }
    .st-emotion-cache-1uj8q5y, .st-emotion-cache-1cynk2p { 
        padding-left: 0rem;
        padding-right: 0rem;
    }

    /* Fix for text alignment in metrics (Streamlit's internal divs) */
    div[data-testid="stMetricValue"] {
        font-size: 2.2em; 
        font-weight: 800;
        color: #3498db; 
        margin-bottom: 8px;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] p {
        color: #718096;
        font-weight: 700;
        font-size: 13px; 
        text-transform: uppercase;
        letter-spacing: 0.8px;
        text-align: center;
    }
            /* Make the username input wider and larger font */
#username_input input {
    font-size: 1.4em !important;
    padding: 15px !important;
    border-radius: 7px !important;
    width: 100% !important;
    min-width: 50px !important;
    max-width: 100px !important;
    box-sizing: border-box;
}
</style>
""", unsafe_allow_html=True)


# --- Data Setup ---
class TaskManagementSystem:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {}
        self.is_trained = False
        self.evaluation_results = {}
        self.priority_encoder = LabelEncoder() # Initialize LabelEncoder here
        self.category_encoder = LabelEncoder() # Also good practice for categories if you use them in other models that require numerical
    def generate_synthetic_data(self, n=100):
        import random
        categories = ['Development', 'Testing', 'Research', 'DevOps', 'Design']
        priorities = ['High', 'Medium', 'Low']
        users = ['Alice', 'Bob', 'Charlie', 'Diana']
        statuses = ['Open', 'In Progress', 'Completed']
        data = []
        for i in range(n):
            data.append({
                'id': f"TASK-{i+1:04d}",
                'description': f"Task description {i+1}",
                'category': random.choice(categories),
                'priority': random.choice(priorities),
                'assigned_user': random.choice(users),
                'created_date': datetime.now() - timedelta(days=random.randint(0, 30)),
                'due_date': datetime.now() + timedelta(days=random.randint(1, 30)),
                'estimated_hours': round(random.uniform(1, 16), 1),
                'status': random.choice(statuses)
            })
        return pd.DataFrame(data)    

    def preprocess_text(self, text):
        # Lowercase
        text = text.lower()
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    

    def train_models(self, df):
        if 'processed_description' not in df.columns:
            df['processed_description'] = df['description'].apply(self.preprocess_text)
        
        X_text = self.tfidf_vectorizer.fit_transform(df['processed_description'])
        
        df['description_length'] = df['description'].str.len()
        df['word_count'] = df['description'].str.split().str.len()
        df['days_until_due'] = (df['due_date'] - datetime.now()).dt.days.fillna(0).clip(lower=0) # Ensure non-negative
        
        additional_features = df[['description_length', 'word_count', 'days_until_due', 'estimated_hours']].fillna(0)
        
        from scipy.sparse import hstack
        X_combined = hstack([X_text, additional_features.values])
        
        # Train category classification model
        y_category = df['category']
        # It's good practice to encode categories too, if you plan to use models like XGBoost/RF on them
        y_category_encoded = self.category_encoder.fit_transform(y_category) # Encode categories
        X_train_cat, X_test_cat, y_train_cat_encoded, y_test_cat_encoded = train_test_split(X_combined, y_category_encoded, test_size=0.2, random_state=42, stratify=y_category_encoded)
        
        nb_model = MultinomialNB()
        nb_model.fit(X_train_cat, y_train_cat_encoded) # Use encoded labels
        nb_pred_encoded = nb_model.predict(X_test_cat)
        
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train_cat, y_train_cat_encoded) # Use encoded labels
        svm_pred_encoded = svm_model.predict(X_test_cat)
        
        # Train priority prediction model
        y_priority = df['priority']
        y_priority_encoded = self.priority_encoder.fit_transform(y_priority) # Encode priorities
        X_train_pri, X_test_pri, y_train_pri_encoded, y_test_pri_encoded = train_test_split(X_combined, y_priority_encoded, test_size=0.2, random_state=42, stratify=y_priority_encoded)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_pri, y_train_pri_encoded) # Use encoded labels
        rf_pred_encoded = rf_model.predict(X_test_pri)
        
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb_model.fit(X_train_pri, y_train_pri_encoded) # Use encoded labels
        xgb_pred_encoded = xgb_model.predict(X_test_pri)
        
        self.models = {
            'naive_bayes': nb_model,
            'svm': svm_model,
            'random_forest': rf_model,
            'xgboost': xgb_model
        }
        
        self.evaluation_results = {
            'category_classification': {
                'naive_bayes': {'accuracy': accuracy_score(y_test_cat_encoded, nb_pred_encoded), 'report': classification_report(y_test_cat_encoded, nb_pred_encoded, output_dict=True, target_names=self.category_encoder.classes_)}, # Pass original labels for report
                'svm': {'accuracy': accuracy_score(y_test_cat_encoded, svm_pred_encoded), 'report': classification_report(y_test_cat_encoded, svm_pred_encoded, output_dict=True, target_names=self.category_encoder.classes_)} # Pass original labels for report
            },
            'priority_prediction': {
                'random_forest': {'accuracy': accuracy_score(y_test_pri_encoded, rf_pred_encoded), 'report': classification_report(y_test_pri_encoded, rf_pred_encoded, output_dict=True, target_names=self.priority_encoder.classes_)}, # Pass original labels for report
                'xgboost': {'accuracy': accuracy_score(y_test_pri_encoded, xgb_pred_encoded), 'report': classification_report(y_test_pri_encoded, xgb_pred_encoded, output_dict=True, target_names=self.priority_encoder.classes_)} # Pass original labels for report
            }
        }
        self.is_trained = True
        return df    
    def predict_task_properties(self, description, estimated_hours=4.0, due_date_offset_days=7):
        if not self.is_trained:
            return "Development", "Medium" 
        
        processed_desc = self.preprocess_text(description)
        
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or not self.tfidf_vectorizer.vocabulary_:
            return "Development", "Medium"

        desc_tfidf = self.tfidf_vectorizer.transform([processed_desc])
        
        description_length = len(description)
        word_count = len(description.split())
        days_until_due = max(0, (datetime.now().date() + timedelta(days=due_date_offset_days) - datetime.now().date()).days)

        additional_features = np.array([[description_length, word_count, days_until_due, estimated_hours]])
        
        from scipy.sparse import hstack
        X_combined = hstack([desc_tfidf, additional_features])
        
        # Predict numerical labels
        predicted_category_encoded = self.models['svm'].predict(X_combined)[0]
        predicted_priority_encoded = self.models['random_forest'].predict(X_combined)[0]
        
        # Inverse transform to get original string labels
        predicted_category = self.category_encoder.inverse_transform([predicted_category_encoded])[0]
        predicted_priority = self.priority_encoder.inverse_transform([predicted_priority_encoded])[0]
        
        return predicted_category, predicted_priority

    def assign_task_to_user(self, task_df, new_task_estimated_hours, excluded_user=None):
        users = task_df['assigned_user'].unique()
        if not list(users):
            return "Unassigned"

        user_workload = task_df.groupby('assigned_user')['estimated_hours'].sum().to_dict()
        for user in users:
            user_workload.setdefault(user, 0)

        # Exclude a user if specified (e.g., for reassignment)
        if excluded_user and excluded_user in user_workload:
            temp_workload = user_workload.copy()
            del temp_workload[excluded_user] # Temporarily remove to find next least loaded
            if not temp_workload: # If only one user, and they are excluded
                return excluded_user # Reassign to self if no other options

            least_loaded_user = min(temp_workload, key=temp_workload.get)
            return least_loaded_user if temp_workload[least_loaded_user] < (user_workload.get(excluded_user, 0) + new_task_estimated_hours) else excluded_user # Only reassign if significantly less loaded
        
        # General assignment to least loaded user
        least_loaded_user = min(user_workload, key=user_workload.get)
        return least_loaded_user

# --- Streamlit UI Components & Pages ---

# Cache the TaskManagementSystem instance
@st.cache_resource
def get_task_system():
    return TaskManagementSystem()

def display_task_card(task_data, task_system_instance):
    """Displays a single task card with status update."""
    
    # Define classes for task status
    status_class_map = {
        'Open': 'status-open',
        'In Progress': 'status-in-progress',
        'Completed': 'status-completed'
    }
    card_class = f"task-card-streamlit {status_class_map.get(task_data['status'], '')}"
    
    # Priority tag styling
    priority_tag_class = f"priority-tag-{task_data['priority']}"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="task-content">
            <div class="task-title">{task_data['id']} - {task_data['description']}</div>
            <div class="task-details">
                <span class="task-priority">⚡️ {task_data['priority']}</span> | 
                <span class="task-year">🗓️ Due: {task_data['due_date'].strftime('%Y-%m-%d')}</span> |
                <span class="task-hours">⏰ {task_data['estimated_hours']} hrs</span>
            </div>
            <div class="task-categories">
                <span class="category-tag">📂 {task_data['category']}</span>
                <span class="{priority_tag_class}"> {task_data['priority']} Priority</span>
                <span class="category-tag">👤 {task_data['assigned_user']}</span>
            </div>
        </div>
        <div class="status-section">
    """, unsafe_allow_html=True) # Open the div for Streamlit controls

    # Streamlit selectbox for status update
    current_status = task_data['status']
    new_status = st.selectbox(
        "Update Status:",
        ['Open', 'In Progress', 'Completed'],
        index=['Open', 'In Progress', 'Completed'].index(current_status),
        key=f"status_select_{task_data['id']}",
        label_visibility="collapsed" # Hide default label for compact UI
    )

    if new_status != current_status:
        # Update status in session state
        idx = st.session_state.data.index[st.session_state.data['id'] == task_data['id']].tolist()[0]
        st.session_state.data.loc[idx, 'status'] = new_status
        st.rerun() # Rerun to update UI with new status

    st.markdown(f"""
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_login_page():
    st.markdown('<div class="header"><h1>🤖 AI-Powered Task Management System</h1><p>Intelligently classify, prioritize, and assign tasks to optimize your workflow</p></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card"><h4 style="margin-bottom:12px;"><span class="icon">👤</span>Login</h4>', unsafe_allow_html=True)
        username = st.text_input("Enter your name to access the system...", key="username_input")
        if st.button("Access System"):
            if username:
                st.session_state.current_user = username
                st.rerun()
            else:
                st.error("Please enter your name!")
        st.markdown('</div>', unsafe_allow_html=True)


def show_main_app_pages(task_system_instance, df):
    # Main layout columns (now with a sidebar equivalent through st.radio)
    st.markdown('<div class="main-layout">', unsafe_allow_html=True) # Open main-layout div

    # Sidebar / Navigation on the left within the first column
    with st.sidebar: # This places elements in the Streamlit sidebar
        st.title("🎛️ Control Panel")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "➕ Add New Task", "📝 Task List", "👥 Workload Management", "📈 Model Performance"],
            index=st.session_state.get('current_page_idx', 0), # Maintain page state
            key="main_navigation_radio",
            format_func=lambda x: x.split(' ')[1] if ' ' in x else x # Just show the name
        )
        st.session_state.current_page_idx = ["📊 Dashboard", "➕ Add New Task", "📝 Task List", "👥 Workload Management", "📈 Model Performance"].index(page)

        st.markdown("---")
        st.info("Welcome to the AI Task Management System! Use the navigation to explore features. For help, contact the admin.", icon="ℹ️")

    # Main content area based on selected page
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "➕ Add New Task":
        show_add_new_task(df, task_system_instance)
    elif page == "📝 Task List":
        show_task_list(df, task_system_instance)
    elif page == "👥 Workload Management":
        show_workload_management(df, task_system_instance)
    elif page == "📈 Model Performance":
        show_model_performance(task_system_instance)

    st.markdown('</div>', unsafe_allow_html=True) # Close main-layout div (this assumes Streamlit's container for columns matches this logic)


def show_dashboard(df):
    st.markdown('<div class="card"><h2><span class="icon">📊</span>Dashboard Overview</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="position:relative; margin-bottom:18px;">
    <div style="position:absolute; right:0; top:-60px; z-index:10;">
        <span style="background:linear-gradient(90deg,#43cea2,#185a9d);color:white;padding:10px 22px;border-radius:20px;font-weight:700;box-shadow:0 2px 8px rgba(67,206,162,0.15);font-size:1.1em;animation:floatAI 2.5s infinite alternate;">
        🤖 Hi, I'm your AI assistant! Ask me to auto-prioritize, summarize, or suggest next actions for your tasks.
        </span>
    </div>
    </div>
<style>
@keyframes floatAI {
  from { transform: translateY(0);}
  to { transform: translateY(-8px);}
}
</style>
""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_tasks = len(df)
    pending_tasks = len(df[df['status'].isin(['Open', 'In Progress'])])
    completed_tasks = len(df[df['status'] == 'Completed'])
    completion_rate = round((completed_tasks / total_tasks) * 100, 1) if total_tasks > 0 else 0
    high_priority_tasks = len(df[df['priority'] == 'High'])

    with col1:
        st.markdown(f"""
        <div class="metric-card-streamlit">
            <div class="metric-value">{total_tasks}</div>
            <div class="metric-label">Total Tasks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card-streamlit">
            <div class="metric-value">{pending_tasks}</div>
            <div class="metric-label">Pending Tasks</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card-streamlit">
            <div class="metric-value">{completion_rate}%</div>
            <div class="metric-label">Completion Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card-streamlit">
            <div class="metric-value">{high_priority_tasks}</div>
            <div class="metric-label">High Priority</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_category = px.pie(
            df, names='category', title='Task Distribution by Category',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_category.update_layout(height=400, title_x=0.5)
        st.plotly_chart(fig_category, use_container_width=True)
    
    with chart_col2:
        priority_order = ['High', 'Medium', 'Low']
        priority_counts = df['priority'].value_counts().reindex(priority_order).fillna(0)
        fig_priority = px.bar(
            x=priority_counts.index, y=priority_counts.values,
            title='Task Priority Distribution',
            color=priority_counts.index,
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f1c40f', 'Low': '#2ecc71'}
        )
        fig_priority.update_layout(height=400, xaxis_title='Priority', yaxis_title='Number of Tasks', title_x=0.5)
        st.plotly_chart(fig_priority, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close card div
    st.markdown("### 🤖 AI Insights")
    urgent_tasks = df[(df['priority'] == 'High') & (df['status'] != 'Completed')]
    if not urgent_tasks.empty:
        st.warning(f"⚡ {len(urgent_tasks)} high-priority tasks need your attention!")
    else:
        st.success("No urgent tasks at the moment. Great job!")


def show_add_new_task(df, task_system_instance):
    st.markdown('<div class="card"><h2><span class="icon">➕</span>Add New Task</h2>', unsafe_allow_html=True)
    st.write("Enter task details below. AI will predict category, priority, and assign a user.")

    with st.form("add_task_form"):
        task_description = st.text_area(
            "Task Description*",
            placeholder="e.g., 'Develop new user registration module with email verification and database integration.'",
            height=150
        )
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            estimated_hours = st.number_input("Estimated Hours*", min_value=0.5, max_value=160.0, value=8.0, step=0.5)
            due_date = st.date_input("Due Date*", value=datetime.now().date() + timedelta(days=7))
        with col_input2:
            manual_override = st.checkbox("Manually Override AI Predictions")
            if manual_override:
                manual_category = st.selectbox("Manual Category", df['category'].unique())
                manual_priority = st.selectbox("Manual Priority", ['High', 'Medium', 'Low'])
            else:
                manual_category = None
                manual_priority = None

        submitted = st.form_submit_button("🤖 Predict & Add Task", type="primary")

        if submitted:
            if not task_description:
                st.error("Please enter a task description.")
            else:
                with st.spinner("Predicting and adding task..."):
                    predicted_category, predicted_priority = "N/A", "N/A"
                    suggested_user = "Unassigned"

                    if task_system_instance.is_trained:
                        if not manual_override:
                            predicted_category, predicted_priority = task_system_instance.predict_task_properties(
                                task_description,
                                estimated_hours=estimated_hours,
                                due_date_offset_days=(due_date - datetime.now().date()).days
                            )
                        else:
                            predicted_category, predicted_priority = manual_category, manual_priority
                        
                        # AI-driven user assignment
                        suggested_user = task_system_instance.assign_task_to_user(
                            st.session_state.data, estimated_hours
                        )
                    else:
                        st.warning("Models not trained. Using default values.")
                        predicted_category = "Development"
                        predicted_priority = "Medium"
                        suggested_user = df['assigned_user'].unique()[0] if not df.empty else "Default User"

                    new_task_id = f"TASK-{len(df) + 1:04d}"
                    new_task_row = pd.DataFrame([{
                        'id': new_task_id,
                        'description': task_description,
                        'category': predicted_category,
                        'priority': predicted_priority,
                        'assigned_user': suggested_user,
                        'created_date': datetime.now(),
                        'due_date': datetime.combine(due_date, datetime.min.time()),
                        'estimated_hours': estimated_hours,
                        'status': 'Open'
                    }])

                    st.session_state.data = pd.concat([st.session_state.data, new_task_row], ignore_index=True)
                    st.success("✅ Task added successfully!")

                    st.subheader("Newly Added Task Details:")
                    col_res1, col_res2, col_res3 = st.columns(3)
                    with col_res1: st.info(f"**Task ID:** {new_task_id}")
                    with col_res2: st.info(f"**Predicted Category:** {predicted_category}")
                    with col_res3: st.info(f"**Predicted Priority:** {predicted_priority}")
                    st.info(f"**Assigned To:** {suggested_user}")
                    
                    st.dataframe(new_task_row[['id', 'description', 'category', 'priority', 'assigned_user', 'due_date', 'estimated_hours', 'status']], use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


def show_task_list(df, task_system_instance):
    st.markdown('<div class="card"><h2><span class="icon">📝</span>Task List</h2>', unsafe_allow_html=True)
    
    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        selected_category = st.selectbox("Filter by Category:", ['All'] + sorted(df['category'].unique().tolist()))
    with col_filter2:
        selected_priority = st.selectbox("Filter by Priority:", ['All'] + ['High', 'Medium', 'Low'])
    with col_filter3:
        selected_status = st.selectbox("Filter by Status:", ['All', 'Open', 'In Progress', 'Completed'])

    filtered_df = df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_priority != 'All':
        filtered_df = filtered_df[filtered_df['priority'] == selected_priority]
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]

    if filtered_df.empty:
        st.markdown(f"""
            <div class="empty-state-streamlit">
                <div class="empty-icon">🤷</div>
                <p>No tasks found matching your filters.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Sort tasks: High priority first, then by due date (soonest first)
        filtered_df['priority_order'] = filtered_df['priority'].map({'High': 0, 'Medium': 1, 'Low': 2})
        filtered_df = filtered_df.sort_values(by=['priority_order', 'due_date'], ascending=[True, True])
        filtered_df = filtered_df.drop(columns=['priority_order'])

        # Display tasks in a grid, respecting screen width
        if st.session_state.get('width', 1200) < 768:
            cols_per_row = 1
        else:
            cols_per_row = 2 
            
        num_rows = (len(filtered_df) + cols_per_row - 1) // cols_per_row
        
        st.markdown('<div style="max-height: 700px; overflow-y: auto; padding-right: 10px;">', unsafe_allow_html=True)
        for i in range(num_rows):
            row_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(filtered_df):
                    task = filtered_df.iloc[idx]
                    with row_cols[j]:
                        display_task_card(task, task_system_instance)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close card div


def show_workload_management(df, task_system_instance):
    st.markdown('<div class="card"><h2><span class="icon">👥</span>Workload Management</h2>', unsafe_allow_html=True)
    st.write("Analyze user workloads and reassign tasks.")

    if df.empty:
        st.markdown(f"""
            <div class="empty-state-streamlit">
                <div class="empty-icon">😔</div>
                <p>No tasks yet to manage workloads. Add some tasks first!</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Workload by user chart
    user_workload = df.groupby('assigned_user')['estimated_hours'].sum().sort_values(ascending=True)
    fig_workload = px.bar(
        x=user_workload.values, y=user_workload.index,
        orientation='h',
        title='Total Estimated Hours by User',
        labels={'x': 'Estimated Hours', 'y': 'User'},
        color=user_workload.values,
        color_continuous_scale='Blues'
    )
    fig_workload.update_layout(title_x=0.5, height=400)
    st.plotly_chart(fig_workload, use_container_width=True)

    # Task Reassignment
    st.subheader("🔄 Reassign Task")
    tasks_to_reassign = df[df['status'] != 'Completed']
    if not tasks_to_reassign.empty:
        task_ids = tasks_to_reassign['id'].tolist()
        task_display_names = [f"{t['id']} - {t['description'][:50]}..." for idx, t in tasks_to_reassign.iterrows()]
        
        selected_task_id = st.selectbox(
            "Select Task to Reassign:",
            options=task_ids,
            format_func=lambda x: next((d for d in task_display_names if x in d), x)
        )

        current_assignee = df[df['id'] == selected_task_id]['assigned_user'].iloc[0]
        st.info(f"Task **{selected_task_id}** is currently assigned to **{current_assignee}**.")

        available_users = df['assigned_user'].unique().tolist()
        new_assignee_options = [user for user in available_users if user != current_assignee]
        if new_assignee_options:
            new_assignee = st.selectbox("Reassign To:", options=new_assignee_options, key="new_assignee_select")
            if st.button("Reassign Task"):
                idx = st.session_state.data.index[st.session_state.data['id'] == selected_task_id].tolist()[0]
                st.session_state.data.loc[idx, 'assigned_user'] = new_assignee
                st.success(f"Task **{selected_task_id}** reassigned to **{new_assignee}**.")
                st.rerun()
        else:
            st.warning("No other users available for reassignment.")
    else:
        st.info("No pending or in-progress tasks to reassign.")

    st.markdown('</div>', unsafe_allow_html=True)


def show_model_performance(task_system_instance):
    st.markdown('<div class="card"><h2><span class="icon">📈</span>Model Performance</h2>', unsafe_allow_html=True)
    
    if not task_system_instance.is_trained:
        st.markdown(f"""
            <div class="empty-state-streamlit">
                <div class="empty-icon">⏳</div>
                <p>Models are still training or not available. Please wait.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.subheader("Categorization Model Performance (Naive Bayes vs. SVM)")
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        st.markdown("**Naive Bayes**")
        nb_metrics = task_system_instance.evaluation_results['category_classification']['naive_bayes']
        st.json(nb_metrics['report'])
    with cat_col2:
        st.markdown("**SVM**")
        svm_metrics = task_system_instance.evaluation_results['category_classification']['svm']
        st.json(svm_metrics['report'])

    st.subheader("Priority Prediction Model Performance (Random Forest vs. XGBoost)")
    pri_col1, pri_col2 = st.columns(2)
    with pri_col1:
        st.markdown("**Random Forest**")
        rf_metrics = task_system_instance.evaluation_results['priority_prediction']['random_forest']
        st.json(rf_metrics['report'])
    with pri_col2:
        st.markdown("**XGBoost**")
        xgb_metrics = task_system_instance.evaluation_results['priority_prediction']['xgboost']
        st.json(xgb_metrics['report'])

    # Comparison charts
    st.subheader("Model Performance Comparison")
    metrics_data = []
    for task_type, models_data in task_system_instance.evaluation_results.items():
        for model_name, metrics in models_data.items():
            metrics_data.append({
                'Model': f"{model_name.replace('_', ' ').title()} ({task_type.replace('_', ' ').title()})",
                'Accuracy': metrics['accuracy']
            })
            # Add other relevant metrics if desired, e.g., Precision, Recall, F1
    
    if metrics_data:
        chart_df = pd.DataFrame(metrics_data)
        fig = px.bar(
            chart_df,
            x='Model',
            y='Accuracy',
            color='Model',
            title='Model Accuracy Comparison',
            height=400,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# --- Main App Logic ---
def main():
    # Initialize session state variables
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'data' not in st.session_state:
        st.session_state.data = None # Will be DataFrame of tasks
    if 'current_page_idx' not in st.session_state:
        st.session_state.current_page_idx = 0 # Default to Dashboard
    # For responsive column sizing (approximate, as Streamlit doesn't expose browser width directly)
    if 'width' not in st.session_state: 
        st.session_state.width = 1200 

    # Get the cached TaskManagementSystem instance
    task_system = get_task_system()

    # If data is not yet loaded and models not trained, do it now
    if st.session_state.data is None:
        with st.spinner("🚀 Initializing AI System: Generating synthetic tasks and training models... This might take a moment!"):
            st.session_state.data = task_system.generate_synthetic_data(1000)
            # Make a copy for training to avoid modifying the original df during feature engineering
            train_df = st.session_state.data.copy()
            task_system.train_models(train_df)
        st.success("✅ AI System ready! Models trained and data loaded.")
        st.rerun() # Rerun to remove spinner and show main content

    # Update user stats text (relevant for Dashboard/Profile in the first column)
    total_tasks_count = len(st.session_state.data)
    pending_tasks_count = len(st.session_state.data[st.session_state.data['status'].isin(['Open', 'In Progress'])])
    st.session_state.user_stats_text = (f"{total_tasks_count} tasks • {pending_tasks_count} pending")

    # Conditional display based on login status
    if st.session_state.current_user is None:
        show_login_page()
    else:
        show_main_app_pages(task_system, st.session_state.data)

if __name__ == "__main__":
    main()
