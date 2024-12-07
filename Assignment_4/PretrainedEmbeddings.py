import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the simplified version of GoEmotions
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# Access train, validation, and test splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]
# Convert to pandas DataFrames
train_df = train_data.to_pandas()
validation_df = validation_data.to_pandas()
test_df = test_data.to_pandas()

def extract_label(label_list):
  return label_list[0]

# Filter rows with exactly one label and extract the single label
train_df = train_df[train_df['labels'].apply(len) == 1]
train_df.loc[:, 'labels'] = train_df['labels'].apply(extract_label)

validation_df = validation_df[validation_df['labels'].apply(len) == 1]
validation_df.loc[:, 'labels'] = validation_df['labels'].apply(extract_label)

test_df = test_df[test_df['labels'].apply(len) == 1]
test_df.loc[:, 'labels'] = test_df['labels'].apply(extract_label)

# Conversion of label objects to Integers
train_df['labels'] = train_df['labels'].astype(int)
validation_df['labels'] = validation_df['labels'].astype(int)
test_df['labels'] = test_df['labels'].astype(int)

# Check shapes after filtering
print("Filtered Training Data Shape:", train_df.shape)
print("\nFiltered Validation Data Shape:", validation_df.shape)
print("\nFiltered Testing Data Shape:", test_df.shape)

# Verify the changes
print("\nSample from Training Data:\n", train_df.head())

print("\nSample from Validation Data:\n", validation_df.head())

print("\nSample from Test Data:\n", test_df.head())

#########################################
import pickle
from wikipedia2vec import Wikipedia2Vec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Path to your pretrained embeddings file
embedding_file = "enwiki_20180420_100d.pkl"

# Load embeddings
print("Loading pretrained embeddings...")
embeddings = Wikipedia2Vec.load(embedding_file)
print("Embeddings loaded successfully!")


def create_document_embeddings(texts, embeddings):
    """
    Create document embeddings by averaging word embeddings for each document.

    Parameters:
        texts (pd.Series): A Pandas Series of text data.
        embeddings (Wikipedia2Vec): Pretrained word embeddings model.

    Returns:
        np.ndarray: An array of document embeddings (num_samples, embedding_dim).
    """
    document_embeddings = []
    for text in texts:
        # Tokenize the text into words
        tokens = text.split()  # Simple tokenization; replace with a better tokenizer if needed

        # Get word vectors for the tokens present in the embeddings
        word_vectors = []
        for word in tokens:
            try:
                vector = embeddings.get_word_vector(word)
                word_vectors.append(vector)
            except KeyError:
                pass  # Skip words not in the vocabulary

        # Compute the document embedding (average word embeddings)
        if word_vectors:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(100)  # Fallback for empty embeddings (100-dimensional here)

        document_embeddings.append(doc_vector)

    return np.array(document_embeddings)


print("Creating document embeddings...")

train_embeddings = create_document_embeddings(train_df['text'], embeddings)
validation_embeddings = create_document_embeddings(validation_df['text'], embeddings)
test_embeddings = create_document_embeddings(test_df['text'], embeddings)

print("Document embeddings created successfully!")

print("Train Embeddings Shape:", train_embeddings.shape)  # (num_train_samples, 100)
print("Validation Embeddings Shape:", validation_embeddings.shape)
print("Test Embeddings Shape:", test_embeddings.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Extract labels
y_train = train_df['labels']
y_validation = validation_df['labels']
y_test = test_df['labels']


# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    # Evaluate on validation data
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"{model_name} Validation Accuracy: {val_accuracy}")

    # Evaluate on test data
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"{model_name} Test Accuracy: {test_accuracy}")
    return model


# Train and evaluate Logistic Regression (Simple Regression)
sr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
train_and_evaluate_model(sr_model, "Simple Regression (Logistic Regression) with class weights", train_embeddings, y_train,
                         validation_embeddings, y_validation, test_embeddings, y_test)

sr_model_unbalanced = LogisticRegression(max_iter=1000, random_state=42)
train_and_evaluate_model(sr_model_unbalanced, "Simple Regression (Logistic Regression) without class weights", train_embeddings, y_train,
                         validation_embeddings, y_validation, test_embeddings, y_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
train_and_evaluate_model(rf_model, "Random Forest without class weights", train_embeddings, y_train, validation_embeddings, y_validation, test_embeddings, y_test)

rf_model_unbalanced = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate_model(rf_model_unbalanced, "Random Forest with class weights", train_embeddings, y_train, validation_embeddings, y_validation, test_embeddings, y_test)

