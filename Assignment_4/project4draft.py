import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer

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

# Inspect the DataFrame
print("Training Data Info:")
train_df.info()
print("\nSample Data:", train_df.head(), "\n\n\n")

print("Validation Data Info:")
validation_df.info()
print("\nSample Data:", validation_df.head(), "\n\n\n")

print("Testing Data Info:")
test_df.info()
print("\nSample Data:", test_df.head(), "\n\n\n")

# Check for null values
print("Missing Values:")
print("\nTraining data frame: \n", train_df.isnull().sum())
print("\nValidation data frame: \n", train_df.isnull().sum())
print("\nTesting data frame: \n", train_df.isnull().sum())

# Delete multiple labels

# Define a helper function to extract the single label
def extract_label(label_list):
  return label_list[0]

# Filter rows with exactly one label and extract the single label
train_df = train_df[train_df['labels'].apply(len) == 1]
train_df.loc[:, 'labels'] = train_df['labels'].apply(extract_label)

validation_df = validation_df[validation_df['labels'].apply(len) == 1]
validation_df.loc[:, 'labels'] = validation_df['labels'].apply(extract_label)

test_df = test_df[test_df['labels'].apply(len) == 1]
test_df.loc[:, 'labels'] = test_df['labels'].apply(extract_label)

# Check shapes after filtering
print("Filtered Training Data Shape:", train_df.shape)
print("\nFiltered Validation Data Shape:", validation_df.shape)
print("\nFiltered Testing Data Shape:", test_df.shape)

# Verify the changes
print("\nSample from Training Data:\n", train_df.head())

print("\nSample from Validation Data:\n", validation_df.head())

print("\nSample from Test Data:\n", test_df.head())

print(train_df['labels'].unique())  # Show all unique values in the column
print(train_df['labels'].dtype)

# Conversion of label objects to Integers
train_df['labels'] = train_df['labels'].astype(int)
validation_df['labels'] = validation_df['labels'].astype(int)
test_df['labels'] = test_df['labels'].astype(int)

# Confirm the data type
print(train_df['labels'].dtype)  # Should now show 'int64'

# Final check
# Check shapes after filtering
print("Filtered Training Data Shape:", train_df.shape)
print("\nFiltered Validation Data Shape:", validation_df.shape)
print("\nFiltered Testing Data Shape:", test_df.shape)

# Verify the changes
print("\nSample from Training Data:\n", train_df.head())

print("\nSample from Validation Data:\n", validation_df.head())

print("\nSample from Test Data:\n", test_df.head())

# Preprocessing for Baseline

# Initialize a Vectorizer
#vectorizer_baseline = CountVectorizer()
vectorizer_baseline = TfidfVectorizer()

# Fit the vectorizer on training data and transform all splits
X_train_baseline = vectorizer_baseline.fit_transform(train_df['text'])
X_validation_baseline = vectorizer_baseline.transform(validation_df['text'])
X_test_baseline = vectorizer_baseline.transform(test_df['text'])

# Labels (already preprocessed)
y_train_baseline = train_df['labels'].to_numpy()  # Convert to NumPy array
y_validation_baseline = validation_df['labels'].to_numpy()
y_test_baseline = test_df['labels'].to_numpy()

# Verify feature matrix shapes
print("Baseline Training Feature Matrix Shape:", X_train_baseline.shape)
print("Baseline Validation Feature Matrix Shape:", X_validation_baseline.shape)
print("Baseline Testing Feature Matrix Shape:", X_test_baseline.shape)
print("Baseline Training label Matrix Shape:", y_train_baseline.shape)
print("Baseline Validation label Matrix Shape:", y_validation_baseline.shape)
print("Baseline Testing label Matrix Shape:", y_test_baseline.shape)

# Preprocessing for Bayesian

# Initialize a Vectorizer
vectorizer_bayesian = CountVectorizer()

# Fit the vectorizer on training data and transform all splits
X_train_bayesian = vectorizer_bayesian.fit_transform(train_df['text'])
X_validation_bayesian = vectorizer_bayesian.transform(validation_df['text'])
X_test_bayesian = vectorizer_bayesian.transform(test_df['text'])

# Labels (already preprocessed)
y_train_bayesian = train_df['labels'].to_numpy()  # Convert to NumPy array
y_validation_bayesian = validation_df['labels'].to_numpy()
y_test_bayesian = test_df['labels'].to_numpy()

# Verify feature matrix shapes
print("Naive Bayes Training Feature Matrix Shape:", X_train_bayesian.shape)
print("Naive Bayes Validation Feature Matrix Shape:", X_validation_bayesian.shape)
print("Naive Bayes Testing Feature Matrix Shape:", X_test_bayesian.shape)
print("Naive Bayes Training label Matrix Shape:", y_train_bayesian.shape)
print("Naive Bayes Validation label Matrix Shape:", y_validation_bayesian.shape)
print("Naive Bayes Testing label Matrix Shape:", y_test_bayesian.shape)

# Class distributions

def class_distribtion(label_set):
  labels, counts = np.unique(label_set, return_counts=True)

  # Calculate frequencies as percentages
  frequencies = (counts / counts.sum()) * 100

  # Create a DataFrame to store counts and frequencies
  class_distribution_table = pd.DataFrame({
    "Class Label": labels,
    "Count": counts,
    "Frequency (%)": frequencies
  })

  # Sort the table by class label for better organization
  class_distribution_table = class_distribution_table.sort_values(by="Class Label").reset_index(drop=True)
  return class_distribution_table

print("Training Label Distribution: \n", class_distribtion(y_train_bayesian))
print("\nValidation Label Distribution:\n", class_distribtion(y_validation_bayesian))
print("\nTesting Label Distribution:\n", class_distribtion(y_test_bayesian))

# Preprocessing for LLM

# Initialize a tokenizer
tokenizer_llm = AutoTokenizer.from_pretrained("bert-base-uncased") # We will use bert-base-uncased

# Tokenize text data for all splits
train_encodings = tokenizer_llm(
    list(train_df['text']), truncation=True, padding='max_length', max_length=128
)
validation_encodings = tokenizer_llm(
    list(validation_df['text']), truncation=True, padding='max_length', max_length=128
)
test_encodings = tokenizer_llm(
    list(test_df['text']), truncation=True, padding='max_length', max_length=128
)

# Convert tokenized data into PyTorch tensors
X_train_llm = torch.tensor(train_encodings['input_ids'])
X_train_attention_masks = torch.tensor(train_encodings['attention_mask'])
y_train_llm = torch.tensor(train_df['labels'].to_numpy())

X_validation_llm = torch.tensor(validation_encodings['input_ids'])
X_validation_attention_masks = torch.tensor(validation_encodings['attention_mask'])
y_validation_llm = torch.tensor(validation_df['labels'].to_numpy())

X_test_llm = torch.tensor(test_encodings['input_ids'])
X_test_attention_masks = torch.tensor(test_encodings['attention_mask'])
y_test_llm = torch.tensor(test_df['labels'].to_numpy())

# Check shapes
print("Training Input IDs Shape:", X_train_llm.shape)
print("Training Attention Masks Shape:", X_train_attention_masks.shape)
print("Training Labels Shape:", y_train_llm.shape)

print("Validation Input IDs Shape:", X_validation_llm.shape)
print("Validation Attention Masks Shape:", X_validation_attention_masks.shape)
print("Validation Labels Shape:", y_validation_llm.shape)

print("Testing Input IDs Shape:", X_test_llm.shape)
print("Testing Attention Masks Shape:", X_test_attention_masks.shape)
print("Testing Labels Shape:", y_test_llm.shape)