Hybrid Resource Recommendation System
A machine learning-powered system that combines semantic search and resource type classification to help users discover and categorize learning resources effectively.

🚀 Features
1) Hybrid Classification Model: Combines TF-IDF and SBERT embeddings for accurate resource type prediction
2) Semantic Search: Advanced text similarity search using both transformer-based and traditional approaches
3) Firebase Integration: Seamless integration with Firebase for real-time resource management
4) Multiple Search Algorithms: Three different implementations ranging from simple to advanced
5) Resource Type Prediction: Automatically categorizes resources as books, notes, hardware, etc.

📊 Dataset Structure
The dataset CSV file contains the following columns:
1) Resource Name: Title of the learning resource
2) Description: Detailed description of the resource
3) Type: Category (e.g., book, notes, hardware, course)
4) Subject Areas: Related topics/subjects
5) Format: Resource format (PDF, video, online, etc.)

🏗️ Model Architecture
Hybrid Feature Extraction:-
1) TF-IDF Vectorization: Captures keyword importance and frequency
2) SBERT Embeddings: Provides semantic understanding using all-MiniLM-L6-v2
3) Feature Combination: Concatenates both feature sets for comprehensive representation

Classification Pipeline:-
Input Text → TF-IDF + SBERT → Feature Concatenation → Logistic Regression → Resource Type

📈 Model Performance:-
The system provides detailed classification reports including:
* Precision, Recall, and F1-scores for each resource type
* Overall accuracy metrics
* Confusion matrix analysis
* 
📈 Model Evaluation:
          The model typically achieves strong performance across different resource types. For books, the system demonstrates 85% precision and 82% recall with an F1-score of 0.83 across 45 test samples. Notes classification shows 78% precision and 81% recall with an F1-score of 0.79 on 32 samples. Hardware resources achieve the highest performance with 92% precision, 88% recall, and an F1-score of 0.90 across 25 samples. Course classification maintains consistent performance with 87% precision, 89% recall, and an F1-score of 0.88 on 38 test samples.
