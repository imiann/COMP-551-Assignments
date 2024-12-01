import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from tqdm import tqdm

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

# Convert labels to integers
train_df['labels'] = train_df['labels'].astype(int)
validation_df['labels'] = validation_df['labels'].astype(int)
test_df['labels'] = test_df['labels'].astype(int)

# Define tokenization function
def tokenize_data(tokenizer, texts, max_length=128):
    return tokenizer(
        list(texts), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt'
    )

# Define models and tokenizers
models_to_evaluate = {
    "bert-base-uncased": "bert-base-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
    "gpt2": "gpt2"
}

# Load tokenizers
tokenizers = {name: AutoTokenizer.from_pretrained(model) for name, model in models_to_evaluate.items()}

# Add padding token for tokenizers without one
for name, tokenizer in tokenizers.items():
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

# Tokenize data for all models
tokenized_data = {}
for name, tokenizer in tokenizers.items():
    print(f"Tokenizing data for {name}...")
    tokenized_data[name] = {
        "train": tokenize_data(tokenizer, train_df['text']),
        "validation": tokenize_data(tokenizer, validation_df['text']),
        "test": tokenize_data(tokenizer, test_df['text'])
    }

# Labels (same across models)
y_train = torch.tensor(train_df['labels'].to_numpy())
y_validation = torch.tensor(validation_df['labels'].to_numpy())
y_test = torch.tensor(test_df['labels'].to_numpy())

# Define evaluation function
def evaluate_model(model, data, labels, batch_size=32):
    dataset = TensorDataset(data['input_ids'], data['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, masks, targets = batch
            outputs = model(input_ids=inputs, attention_mask=masks)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Evaluate all models
results = {}
for name, model_name in models_to_evaluate.items():
    print(f"Evaluating {name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=28)
    acc = evaluate_model(model, tokenized_data[name]["test"], y_test)
    results[name] = acc
    print(f"Accuracy for {name}: {acc}")

# Display results
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
results_df.index.name = "Model"
results_df.reset_index(inplace=True)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy of Pretrained Models on GoEmotions")
plt.xticks(rotation=45)
plt.show()
