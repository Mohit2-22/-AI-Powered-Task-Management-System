md
🤖 AI-Powered Bug Classification System
A sophisticated task management system that leverages machine learning to automatically classify and prioritize bug reports using historical Eclipse bug data. The system provides real-time predictions and an intuitive Streamlit interface for task management.

🎯 Features
🔍 Intelligent Bug Classification

Automatic product category prediction (e.g., Capella, Platform, Community)
Smart priority level assignment (P1-P5)
Automatic developer assignment based on product category
📊 Task Management

Real-time task analysis and classification
Interactive task submission interface
Comprehensive task history tracking
Task export functionality (CSV format)
🎨 User Interface

Clean, modern Streamlit interface
Intuitive task submission form
Detailed task analysis results
Organized task history view
🛠️ Technologies Used
Programming Language

Python 3.x
Web Framework

Streamlit
Machine Learning

scikit-learn
NLTK (Natural Language Processing)
TF-IDF Vectorization
Naive Bayes Classifier
Random Forest Classifier
Data Processing

pandas
NumPy
joblib (model serialization)
🏗️ Architecture
Data Processing Layer

Text preprocessing using NLTK
Stop word removal and stemming
TF-IDF vectorization for text features
Machine Learning Layer

Product Classification Model (Naive Bayes)
Priority Prediction Model (Random Forest)
Pre-trained models stored as .joblib files
Application Layer

Streamlit web interface
Real-time prediction pipeline
Task history management
Data export functionality
📦 Installation and Setup
Clone the Repository

git clone <repository-url>
cd <project-directory>
Install Dependencies

pip install -r requirements.txt
Run the Application

streamlit run jupyterNotebook/app.py
🤖 Models Used
Product Category Classifier

Model: Multinomial Naive Bayes
Input: Bug description text
Output: Product category prediction
Features: TF-IDF vectorized text (3000 features)
File: classifier_model.joblib
Priority Predictor

Model: Random Forest Classifier
Input: Bug description text
Output: Priority level (P1-P5)
Features: Same TF-IDF vectorization
File: priority_model.joblib
Text Vectorizer

TF-IDF Vectorizer
Max Features: 3000
File: tfidf_vectorizer.joblib
📊 Data Processing Flow
Text Preprocessing

Remove non-alphanumeric characters
Convert to lowercase
Remove stop words
Apply Porter Stemming
Feature Extraction

Convert text to TF-IDF vectors
Use consistent vectorization for both models
Prediction Pipeline

Vectorize input text
Predict product category and priority
Assign developer based on category
Store task in history
🚀 Future Scope
Model Improvements

Implement deep learning models (BERT, RoBERTa)
Add confidence scores for predictions
Support multi-label classification
Feature Enhancements

Add user authentication
Implement task editing and deletion
Add task comments and status updates
Integration with issue tracking systems
Performance Optimization

Model compression for faster inference
Batch processing for multiple tasks
Caching for repeated queries
UI/UX Improvements

Advanced filtering and sorting
Custom themes and layouts
Interactive visualizations
Mobile-responsive design
👥 Development Team
Backend & ML Engineering: [Mohit prajapati]
Data Science: [Mohit prajapati]
UI/UX Design: [Mohit prajapati]
