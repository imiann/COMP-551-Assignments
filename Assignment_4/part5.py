import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load and preprocess the dataset
def load_data():
    dataset = load_dataset("go_emotions")
    
    # Filter single-label examples
    def filter_single_label(example):
        return len(example["labels"]) == 1

    train_data = dataset["train"].filter(filter_single_label)
    val_data = dataset["validation"].filter(filter_single_label)
    test_data = dataset["test"].filter(filter_single_label)

    # Convert labels from lists to integers
    train_data = train_data.map(lambda x: {"labels": x["labels"][0]})
    val_data = val_data.map(lambda x: {"labels": x["labels"][0]})
    test_data = test_data.map(lambda x: {"labels": x["labels"][0]})

    return train_data, val_data, test_data


# Tokenize the dataset
def tokenize_data(dataset, tokenizer, max_length=128):
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )


# Main training function
def train_bert_with_class_weight():
    # Load and preprocess data
    train_data, val_data, test_data = load_data()

    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)
    test_data = tokenize_data(test_data, tokenizer)

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Convert to DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Compute class weights
    labels = np.array(train_data["labels"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=28)

    # Freeze BERT layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer, scheduler, and weighted loss function
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Training loop
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
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_validation_loss = total_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation loss: {avg_validation_loss}, Accuracy: {accuracy}")

        # Save the best model
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), "bert_finetuned_best.pth")

        scheduler.step()

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load("bert_finetuned_best.pth"))
    model.eval()

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


# Run the script
if __name__ == "__main__":
    train_bert_with_class_weight()
