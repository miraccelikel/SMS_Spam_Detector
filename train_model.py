import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Import common functions from utils.py
from utils import clean_text, tokenizer_porter


def train_and_save():
    print("Loading data...")

    # Define file path (Make sure 'Spam_SMS.csv' is in 'data' folder)
    file_path = 'data/Spam_SMS.csv'

    if not os.path.exists(file_path):
        print(f"ERROR: File not found at '{file_path}'. Please check the file name and path.")
        return

    try:
        # header=0 implies the first line is the header ("Class,Message")
        df = pd.read_csv(file_path, sep=',', names=["class", "message"], header=0, encoding='utf-8')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. Preprocessing
    print("Preprocessing data...")
    df.dropna(inplace=True)  # Remove missing values if any
    df['message'] = df['message'].apply(clean_text)
    X = df['message']
    y = df['class']

    # 2. Split Data
    print("Splitting data into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Build Pipeline
    print("Training model... (This might take a moment)")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenizer_porter, stop_words='english', ngram_range=(1, 3))),
        ('clf', SVC(kernel='sigmoid', C=10, probability=True, class_weight='balanced', gamma='scale'))
    ])

    pipeline.fit(X_train, y_train)

    # 4. Evaluate & Save Model
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label='spam')

    print(f"Model Accuracy: {acc * 100:.2f}%")
    print(f"Spam Recall: {rec * 100:.2f}%")

    # Save Model & Metrics
    # Create directory if not exists
    os.makedirs('models', exist_ok=True)

    joblib.dump(pipeline, 'models/spam_model.pkl')
    print("Model saved to 'models/spam_model.pkl'.")

    metrics = {
        "accuracy": acc,
        "recall": rec
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to 'models/metrics.json'.")

    # 5. Generate & Save Confusion Matrix Image
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix (Test Set)')

    # Create directory if not exists
    os.makedirs('assets', exist_ok=True)

    plt.savefig('assets/confusion_matrix.png', bbox_inches='tight')
    print("Matrix image saved to 'assets/confusion_matrix.png'.")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    train_and_save()