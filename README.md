Hybrid Resource Recommendation System
A machine learning-powered system that combines semantic search and resource type classification to help users discover and categorize learning resources effectively.

🚀 Features
1) Hybrid Classification Model: Combines TF-IDF and SBERT embeddings for accurate resource type prediction
2) Semantic Search: Advanced text similarity search using both transformer-based and traditional approaches
3) Firebase Integration: Seamless integration with Firebase for real-time resource management
4) Multiple Search Algorithms: Three different implementations ranging from simple to advanced
5) Resource Type Prediction: Automatically categorizes resources as books, notes, hardware, etc.

📊 Dataset Structure
Your CSV file should contain the following columns:
1) Resource Name: Title of the learning resource
2) Description: Detailed description of the resource
3) Type: Category (e.g., book, notes, hardware, course)
4) Subject Areas: Related topics/subjects
5) Format: Resource format (PDF, video, online, etc.)

🏗️ Model Architecture
Hybrid Feature Extraction:-
TF-IDF Vectorization: Captures keyword importance and frequency
SBERT Embeddings: Provides semantic understanding using all-MiniLM-L6-v2
Feature Combination: Concatenates both feature sets for comprehensive representation

Classification Pipeline:-
Input Text → TF-IDF + SBERT → Feature Concatenation → Logistic Regression → Resource Type

📈 Model Performance:-
The system provides detailed classification reports including:
Precision, Recall, and F1-scores for each resource type
Overall accuracy metrics
Confusion matrix analysis

Example output:
📈 Model Evaluation:
              precision    recall  f1-score   support
       book       0.85      0.82      0.83        45
      notes       0.78      0.81      0.79        32
   hardware       0.92      0.88      0.90        25
     course       0.87      0.89      0.88        38
�
