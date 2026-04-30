import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
import os

DATASET_PATH = "dataset/fake_news_dataset.csv"

if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "dataset/fake_news_dataset_sample.csv"

data = pd.read_csv(DATASET_PATH)


# Combine title and text
data['content'] = data['title'] + " " + data['text']

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

data['content'] = data['content'].apply(preprocess)

# Features and labels
X = data['content']
y = data['label']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model saved successfully")
