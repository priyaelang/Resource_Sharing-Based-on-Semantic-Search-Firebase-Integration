Hybrid Resource Recommendation System
A machine learning-powered system that combines semantic search and resource type classification to help users discover and categorize learning resources effectively.
🚀 Features

Hybrid Classification Model: Combines TF-IDF and SBERT embeddings for accurate resource type prediction
Semantic Search: Advanced text similarity search using both transformer-based and traditional approaches
Firebase Integration: Seamless integration with Firebase for real-time resource management
Multiple Search Algorithms: Three different implementations ranging from simple to advanced
Resource Type Prediction: Automatically categorizes resources as books, notes, hardware, etc.

📋 Requirements
pandas>=1.3.0
numpy>=1.21.0
sentence-transformers>=2.0.0
scikit-learn>=1.0.0
torch>=1.9.0
joblib>=1.0.0
🛠️ Installation

Clone the repository:

bashgit clone <repository-url>
cd hybrid-resource-recommendation

Install dependencies:

bashpip install -r requirements.txt

Download the dataset:

Place your Learning_Resources_Database.csv in the ./dataset/ directory
Ensure the CSV contains columns: Resource Name, Description, Type, Subject Areas, Format



📊 Dataset Structure
Your CSV file should contain the following columns:

Resource Name: Title of the learning resource
Description: Detailed description of the resource
Type: Category (e.g., book, notes, hardware, course)
Subject Areas: Related topics/subjects
Format: Resource format (PDF, video, online, etc.)

🔧 Usage
1. Advanced Hybrid Model (Recommended)
pythonfrom hybrid_resource_model import predict_resource_type

# Predict resource type
result = predict_resource_type(
    "Python Programming Book", 
    "Comprehensive guide on Python for beginners"
)
print(f"Predicted type: {result}")
2. Semantic Search with SBERT
pythonimport pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
import torch

# Load pre-trained model and data
df = joblib.load("resource_data.pkl")
corpus_embeddings = torch.load("corpus_embeddings.pt")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Perform semantic search
query = "machine learning course"
query_embedding = embedder.encode(query, convert_to_tensor=True)
# ... (implement similarity search)
3. Simple Text-Based Search
pythonfrom hybrid_resource_model import semantic_search

# Basic semantic search
semantic_search("data science course", top_n=5)
4. Firebase Integration
pythonfrom hybrid_resource_model import semantic_search_from_firebase

# Search through Firebase resources
firebase_data = {
    "user1": {
        "res1": {
            "name": "Python Basics",
            "description": "Introduction to Python programming",
            "type": "course"
        }
    }
}

results = semantic_search_from_firebase("python programming", firebase_data)
🏗️ Model Architecture
Hybrid Feature Extraction

TF-IDF Vectorization: Captures keyword importance and frequency
SBERT Embeddings: Provides semantic understanding using all-MiniLM-L6-v2
Feature Combination: Concatenates both feature sets for comprehensive representation

Classification Pipeline
Input Text → TF-IDF + SBERT → Feature Concatenation → Logistic Regression → Resource Type
📈 Model Performance
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
🔍 Search Algorithms
1. Advanced SBERT Search

Uses pre-trained transformer models
High semantic understanding
Best for complex queries

2. Hybrid TF-IDF + SBERT

Combines keyword matching with semantic search
Balanced performance and accuracy
Recommended for production use

3. Simple TF-IDF Search

Lightweight and fast
Good for basic keyword matching
Suitable for resource-constrained environments

📁 File Structure
project/
├── hybrid_resource_model.py    # Main model implementations
├── dataset/
│   └── Learning_Resources_Database.csv
├── resource_data.pkl          # Processed resource data
├── corpus_embeddings.pt       # Pre-computed embeddings
├── resource_type_classifier.pkl # Trained classifier
├── requirements.txt
└── README.md
🚀 Getting Started

Train the Model:

pythonpython hybrid_resource_model.py

Make Predictions:

python# Resource type prediction
predict_resource_type("Data Science Handbook", "Complete guide to data science")

# Semantic search
semantic_search("machine learning algorithms")

Integrate with Firebase:

python# Use semantic_search_from_firebase for real-time search
results = semantic_search_from_firebase(query, firebase_resources)
🔧 Configuration
Model Parameters

TF-IDF max_features: 5000 (adjustable)
SBERT model: all-MiniLM-L6-v2 (can be upgraded)
Classifier: Logistic Regression with max_iter=1000
Test split: 20% (configurable)

Performance Tuning

Increase max_features for larger vocabularies
Use larger SBERT models (all-mpnet-base-v2) for better accuracy
Experiment with different classifiers (Random Forest, SVM)

🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🐛 Troubleshooting
Common Issues

Import Errors: Ensure all dependencies are installed
CUDA Issues: The model automatically falls back to CPU if CUDA is unavailable
Memory Issues: Reduce max_features or use smaller SBERT models
Dataset Issues: Verify CSV column names match the expected format

Performance Optimization

Use GPU acceleration for faster SBERT encoding
Cache embeddings for frequently searched resources
Implement batch processing for large datasets

📞 Support
For questions and support, please open an issue on the GitHub repository or contact the development team.

Happy Learning Resource Discovery! 🎓
