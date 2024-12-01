import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load the simplified version of GoEmotions
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# Convert to pandas DataFrames and filter single-label examples
train_data = dataset["train"].to_pandas()
validation_data = dataset["validation"].to_pandas()
test_data = dataset["test"].to_pandas()

# Define a helper function to extract the single label
def extract_label(label_list):
    return label_list[0]

# Filter and process labels
train_data = train_data[train_data['labels'].apply(len) == 1]
train_data['labels'] = train_data['labels'].apply(extract_label).astype(int)

validation_data = validation_data[validation_data['labels'].apply(len) == 1]
validation_data['labels'] = validation_data['labels'].apply(extract_label).astype(int)

test_data = test_data[test_data['labels'].apply(len) == 1]
test_data['labels'] = test_data['labels'].apply(extract_label).astype(int)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
def tokenize_data(data, tokenizer, max_length=128):
    return tokenizer(
        list(data['text']), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt'
    )

train_encodings = tokenize_data(train_data, tokenizer)
validation_encodings = tokenize_data(validation_data, tokenizer)
test_encodings = tokenize_data(test_data, tokenizer)

# Convert labels to PyTorch tensors
y_train = torch.tensor(train_data['labels'].values)
y_validation = torch.tensor(validation_data['labels'].values)
y_test = torch.tensor(test_data['labels'].values)

# Prepare datasets and dataloaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
validation_dataset = TensorDataset(validation_encodings['input_ids'], validation_encodings['attention_mask'], y_validation)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load the pretrained BERT model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)

# Freeze all BERT layers except the classification head
for param in model.base_model.parameters():
    param.requires_grad = False

# Replace the classification head (optional, as it's already initialized)
model.classifier = nn.Linear(model.config.hidden_size, 28)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer, scheduler, and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
loss_fn = nn.CrossEntropyLoss()

# Train the model
num_epochs = 3
best_validation_loss = float("inf")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training phase
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_train_loss}")
    
    # Validation phase
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_validation_loss = total_loss / len(validation_loader)
    accuracy = correct / total
    print(f"Validation loss: {avg_validation_loss}, Accuracy: {accuracy}")
    
    # Save the best model
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        torch.save(model.state_dict(), "bert_finetuned_best.pth")
    
    scheduler.step()

# Load the best model
model.load_state_dict(torch.load("bert_finetuned_best.pth"))
model.eval()

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy}")
