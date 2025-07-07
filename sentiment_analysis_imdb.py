import os
import re
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load IMDb Dataset

def load_imdb_data(directory):
    texts, sentiments = [], []
    for label in ['pos', 'neg']:
        labeled_dir = os.path.join(directory, label)
        for i, filename in enumerate(os.listdir(labeled_dir)):
            if filename.endswith(".txt"):
                with open(os.path.join(labeled_dir, filename), encoding='utf-8') as file:
                    texts.append(file.read())
                    sentiments.append(label)
            if i % 1000 == 0:
                print(f"Loaded {i} files from {label}")
    return pd.DataFrame({"text": texts, "sentiment": sentiments})

# Replace this path with your extracted aclImdb path
train_data = load_imdb_data("./aclImdb/train")
test_data = load_imdb_data("./aclImdb/test")
data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocess text data
data['processed_text'] = data['text'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_text'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr, average='weighted'))
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()

# Save confusion matrix plot as an image
cm_image_path = "confusion_matrix_lr.png"
plt.savefig(cm_image_path)  # Save plot as PNG
print(f"Confusion Matrix Plot Saved at {os.path.abspath(cm_image_path)}")
plt.show()  # Display the plot
plt.close()  # Close the plot to avoid overlapping with the next one

# Precision, Recall, and F1-Score for Logistic Regression
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_lr)

# Create a DataFrame for the metrics
metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}, index=label_encoder.classes_)

# Plot Precision, Recall, F1-Score
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Precision, Recall, and F1-Score for Each Class (Logistic Regression)')
plt.xlabel('Class')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower left')
plt.tight_layout()

# Save the bar plot as an image
metrics_image_path = "precision_recall_f1_lr.png"
plt.savefig(metrics_image_path)  # Save plot as PNG
print(f"Precision-Recall-F1 Plot Saved at {os.path.abspath(metrics_image_path)}")
plt.show()  # Display the plot
plt.close()  # Close the plot
