import os
import torch
from datasets import load_dataset
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Centralized Configuration
SETTINGS = {
    "max_token_length": 128,
    "eval_batch_size": 32,
    "output_directory": "./ResultsPart2",
}

# Ensure the Results directory exists
os.makedirs(SETTINGS["output_directory"], exist_ok=True)

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        label = item["labels"]

        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
            "text": text,  # Include original text for visualization
        }


# Data Loader
def fetch_and_prepare_emotion_data():
    """Load and filter the GoEmotions dataset."""
    dataset = load_dataset("go_emotions")

    # Retain only single-label examples
    def single_label_filter(example):
        return len(example["labels"]) == 1

    train_set = dataset["train"].filter(single_label_filter).select(range(3200))
    val_set = dataset["validation"].filter(single_label_filter).select(range(3200))
    test_set = dataset["test"].filter(single_label_filter).select(range(3200))

    # Simplify label structure
    for split in [train_set, val_set, test_set]:
        split.map(lambda x: {"labels": x["labels"][0]}, batched=False)

    return train_set, val_set, test_set


# Detailed Metrics
def generate_detailed_metrics(model_identifier, true_labels, predicted_labels, label_names, results_dir):
    """Generate and save detailed evaluation metrics."""
    model_results_dir = os.path.join(results_dir, model_identifier)
    os.makedirs(model_results_dir, exist_ok=True)

    # Accuracy and F1-score
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    print(f"Metrics for {model_identifier}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}\n")

    # Confusion Matrix
    cmatrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", cmatrix)

    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=label_names, digits=4)
    print("\nClassification Report:\n", report)

    # Save metrics to files
    with open(os.path.join(model_results_dir, "classification_report.txt"), "w") as report_file:
        report_file.write(report)
    np.savetxt(os.path.join(model_results_dir, "confusion_matrix.csv"), cmatrix, delimiter=",", fmt="%d")

    # Plot confusion matrix
    create_confusion_matrix_plot(cmatrix, label_names, model_identifier, model_results_dir)

    print(f"Metrics and confusion matrix saved in: {model_results_dir}")
    return accuracy, f1


def create_confusion_matrix_plot(matrix, labels, model_identifier, results_dir):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(f"Confusion Matrix: {model_identifier}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], "d"),
                     ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    plt.tight_layout()
    save_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


# Evaluation Loop
def evaluate_model(model, dataloader, device, class_labels, model_identifier, results_dir):
    model.eval()
    true_labels, predicted_labels = [], []
    texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            texts.extend(batch["text"])

    accuracy, f1 = generate_detailed_metrics(
        model_identifier,
        true_labels,
        predicted_labels,
        class_labels,
        results_dir
    )
    return accuracy, f1


# Main Script
def main():
    # Load dataset
    _, _, test_set = fetch_and_prepare_emotion_data()

    # Emotion labels for GoEmotions
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
    ]

    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(emotion_labels))

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare data loaders
    test_dataset = EmotionDataset(test_set, tokenizer, SETTINGS["max_token_length"])
    test_loader = DataLoader(test_dataset, batch_size=SETTINGS["eval_batch_size"])

    # Test Evaluation
    test_accuracy, test_f1 = evaluate_model(
        model,
        test_loader,
        device,
        emotion_labels,
        model_name,
        SETTINGS["output_directory"]
    )
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1-Score: {test_f1:.4f}")


if __name__ == "__main__":
    main()
