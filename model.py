import pandas as pd
import numpy as np
import re
from math import sqrt
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import torch

df = pd.read_csv("./dataset/Learning_Resources_Database.csv")
df = df.dropna(subset=["title", "description", "label"])
X = df["title"] + " " + df["description"]
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sbert_embeddings(texts):
    return sbert_model.encode(texts, convert_to_tensor=True)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()
X_sbert = get_sbert_embeddings(X)
X_combined = np.hstack((X_tfidf, X_sbert.cpu().numpy()))

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def predict_resource_type(title, description):
    text = title + " " + description
    X_tfidf_new = tfidf_vectorizer.transform([text]).toarray()
    X_sbert_new = get_sbert_embeddings([text])
    X_combined_new = np.hstack((X_tfidf_new, X_sbert_new.cpu().numpy()))
    predicted = classifier.predict(X_combined_new)
    predicted_label = label_encoder.inverse_transform(predicted)
    print(predicted_label[0])
    return predicted_label[0]

predict_resource_type("Python Programming Book", "Comprehensive guide on Python for beginners")

df = pd.read_csv("./dataset/Learning_Resources_Database.csv")
df = df.dropna(subset=["Resource Name", "Description", "Type"])
df["text"] = df["Resource Name"] + ". " + df["Description"]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embedder.encode(df["text"].tolist(), convert_to_tensor=True)

joblib.dump(df, "resource_data.pkl")
torch.save(corpus_embeddings, "corpus_embeddings.pt")

X = embedder.encode(df["text"].tolist())
y = df["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump(clf, "resource_type_classifier.pkl")

df = pd.read_csv("./dataset/Learning_Resources_Database.csv")
df["text"] = (
    df["Resource Name"].fillna("") + " " +
    df["Description"].fillna("") + " " +
    df["Subject Areas"].fillna("") + " " +
    df["Type"].fillna("") + " " +
    df["Format"].fillna("")
)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t]

df["tokens"] = df["text"].apply(tokenize)
vocab = set()
for tokens in df["tokens"]:
    vocab.update(tokens)
vocab = sorted(list(vocab))
vocab_index = {word: idx for idx, word in enumerate(vocab)}

def compute_tf_vector(tokens, vocab_index):
    vector = [0] * len(vocab_index)
    for token in tokens:
        if token in vocab_index:
            vector[vocab_index[token]] += 1
    return vector

df["tf_vector"] = df["tokens"].apply(lambda tokens: compute_tf_vector(tokens, vocab_index))

def cosine_similarity(vec1, vec2):
    dot = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = sqrt(sum(x * x for x in vec1))
    norm2 = sqrt(sum(y * y for y in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def semantic_search(query, top_n=5):
    query_tokens = tokenize(query)
    query_vec = compute_tf_vector(query_tokens, vocab_index)
    similarities = []
    for i, row in df.iterrows():
        sim = cosine_similarity(query_vec, row["tf_vector"])
        similarities.append((i, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_n]
    for idx, score in top_results:
        print(f"\nScore: {score:.4f}")
        print("Resource Name:", df.loc[idx, "Resource Name"])
        print("Description:", df.loc[idx, "Description"])
        print("Type:", df.loc[idx, "Type"])
        print("Subject Areas:", df.loc[idx, "Subject Areas"])

semantic_search("data science course")

def semantic_search_from_firebase(query, firebase_resources, top_n=5):
    documents = []
    metadata = []
    for user_id, resources in firebase_resources.items():
        for res_id, resource in resources.items():
            combined_text = f"{resource.get('name', '')} {resource.get('description', '')} {resource.get('type', '')}"
            tokens = tokenize(combined_text)
            documents.append(tokens)
            metadata.append((resource, user_id, res_id))
    vocab = sorted(list(set(token for doc in documents for token in doc)))
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    tf_vectors = [compute_tf_vector(doc, vocab_index) for doc in documents]
    query_tokens = tokenize(query)
    query_vec = compute_tf_vector(query_tokens, vocab_index)
    similarities = []
    for i, tf_vector in enumerate(tf_vectors):
        score = cosine_similarity(query_vec, tf_vector)
        similarities.append((i, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_n]
    results = []
    for idx, score in top_matches:
        res_info, user_id, res_id = metadata[idx]
        res_info["score"] = round(score, 4)
        results.append(res_info)
    return results
