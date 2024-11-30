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


# Conversion of label objects to Integers
train_df['labels'] = train_df['labels'].astype(int)
validation_df['labels'] = validation_df['labels'].astype(int)
test_df['labels'] = test_df['labels'].astype(int)


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




# -----------------------------------

