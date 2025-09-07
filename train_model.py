import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC # NEW IMPORT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# --- NLTK resources and configuration remain the same ---
print("Checking NLTK resources...")
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK resources checked/downloaded.")

INTENTS_FILE = 'intents.json'
MODEL_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'chatbot_model.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()
custom_words_to_keep = {'how', 'are', 'you', 'what', 'is', 'a', 'the', 'hi', 'bye','fees', 'college', 'campus', 'course', 'event', 'hostel', 'library','contact', 'placement', 'faculty', 'sport', 'faq', 'about', 'admission','and', 'or', 'for', 'to', 'in', 'of', 'on', 'with', 'do', 'have', 'i', 's'}
filtered_stop_words = set(w for w in stopwords.words('english') if w not in custom_words_to_keep)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in filtered_stop_words and word.isalpha()]
    return ' '.join(tokens)

# --- Main Training Logic ---
def train_and_save_model():
    print("Loading intents from:", INTENTS_FILE)
    with open(INTENTS_FILE, encoding='utf-8') as file:
        knowledge_base = json.load(file)

    X_patterns = []
    y_labels = []

    for intent in knowledge_base['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            X_patterns.append(pattern)
            y_labels.append(tag)

    if not X_patterns:
        print("Error: No patterns found in intents.json. Cannot train model.")
        return

    print(f"Found {len(X_patterns)} patterns across {len(set(y_labels))} intents.")
    print("Preprocessing text data...")
    X_processed = [preprocess_text(text) for text in X_patterns]
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X_processed)
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # --- NEW: Train LinearSVC Classifier ---
    print("Training Linear Support Vector Classifier (LinearSVC) model...")
    model = LinearSVC()
    model.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("------------------------")

    print(f"Saving model components to {MODEL_DIR}/...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(model, MODEL_PATH)
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_and_save_model()