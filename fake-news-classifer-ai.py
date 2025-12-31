# ----------------------------------------
# Fake News Detection using AI — Colab Code
# ----------------------------------------

# Imports
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Upload the CSV (use Colab file upload)
from google.colab import files
uploaded = files.upload()

# Load dataset (name must match uploaded file)
data = pd.read_csv("fake_real_news_sample.csv")

# Optional: show first few rows
print("Dataset Preview:")
print(data.head())

# Combine title + text into one column (optional but improves model)
data['combined_text'] = data['title'].fillna('') + " " + data['text'].fillna('')

# Ensure text is string
data['combined_text'] = data['combined_text'].astype(str)

# Preprocess text function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove non‑letters
    text = text.lower()
    return text

data['clean_text'] = data['combined_text'].apply(clean_text)

# Features and labels
X = data['clean_text']
y = data['label'].astype(int)

# Train / test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF‑IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Model training completed!")

# Accuracy check
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

# Test on custom news input
test_news = ["Scientists report progress on renewable energy research"]
test_vec = vectorizer.transform(test_news)
prediction = model.predict(test_vec)
print("Custom Test Prediction:", "Fake" if prediction[0]==0 else "Real")

# Test on multiple new samples
samples = [
    "Aliens landed on Earth",
    "New smartphone model released with advanced AI"
]
sample_vec = vectorizer.transform(samples)
preds = model.predict(sample_vec)
for i, s in enumerate(samples):
    print(f"\nNews: {s}")
    print("Predicted:", "Fake" if preds[i]==0 else "Real")
