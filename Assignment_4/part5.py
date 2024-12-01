import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


# Centralized Configuration
SETTINGS = {
    "max_token_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 5e-5,
    "epochs": 3,
    "weight_decay_factor": 0.01,
    "log_interval_steps": 100,
    "output_directory": "./results",
    "log_directory": "./logs",
}


# Data Loader
def fetch_and_prepare_emotion_data():
    """Load and filter the GoEmotions dataset."""
    dataset = load_dataset("go_emotions")

    # Retain only single-label examples
    def single_label_filter(example):
        return len(example["labels"]) == 1

    train_set = dataset["train"].filter(single_label_filter)
    val_set = dataset["validation"].filter(single_label_filter)
    test_set = dataset["test"].filter(single_label_filter)

    # Simplify label structure
    for split in [train_set, val_set, test_set]:
        split.map(lambda x: {"labels": x["labels"][0]}, batched=False)

    return train_set, val_set, test_set


# Tokenization Handler
def tokenize_data(dataset, tokenizer, max_length):
    """Apply tokenization to text data."""
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )


# Class Weights Generator
def compute_weights(labels_array):
    """Generate weights for imbalanced classes."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_array),
        y=labels_array,
    )
    return torch.tensor(weights, dtype=torch.float)


# Metric Computation
def evaluate_predictions(predictions):
    """Calculate metrics for evaluation."""
    true_labels = predictions.label_ids
    predicted_labels = predictions.predictions.argmax(-1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# Visualization and Metrics
def generate_detailed_metrics(model_identifier, true_labels, predicted_labels, label_names):
    """Generate and save detailed evaluation metrics."""
    print(f"Metrics for {model_identifier}:")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"Weighted F1-Score: {f1_score(true_labels, predicted_labels, average='weighted'):.4f}\n")

    cmatrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", cmatrix)

    report = classification_report(true_labels, predicted_labels, target_names=label_names, digits=4)
    print("\nClassification Report:\n", report)

    create_confusion_matrix_plot(cmatrix, label_names, model_identifier)


# Confusion Matrix Plotting
def create_confusion_matrix_plot(matrix, labels, model_identifier):
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
    plt.savefig(f"confusion_matrix_{model_identifier.split('/')[-1]}.png")
    plt.close()


# Training and Evaluation Workflow
def execute_model_pipeline(model_identifier, train_set, val_set, test_set, class_labels, config):
    """Train and evaluate the specified model."""
    print(f"Initializing {model_identifier}...")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    # Handle missing padding tokens for certain models (e.g., GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=len(class_labels))

    # Adjust embeddings for new tokens
    if tokenizer.pad_token:
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize datasets
    train_set = tokenize_data(train_set, tokenizer, config["max_token_length"])
    val_set = tokenize_data(val_set, tokenizer, config["max_token_length"])
    test_set = tokenize_data(test_set, tokenizer, config["max_token_length"])

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Compute class weights
    train_labels = np.array(train_set["labels"])
    weights = compute_weights(train_labels).to("cuda" if torch.cuda.is_available() else "cpu")

    # Training configuration
    training_args = TrainingArguments(
        output_dir=f"{config['output_directory']}_{model_identifier.split('/')[-1]}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay_factor"],
        logging_dir=f"{config['log_directory']}_{model_identifier.split('/')[-1]}",
        logging_steps=config["log_interval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        compute_metrics=evaluate_predictions,
    )

    # Training
    print(f"Training {model_identifier}...")
    trainer.train()

    # Evaluation
    print(f"Evaluating {model_identifier}...")
    evaluation_results = trainer.evaluate(test_set)
    print(f"Evaluation Results for {model_identifier}: {evaluation_results}")

    # Generate extended metrics
    predictions = trainer.predict(test_set).predictions.argmax(-1)
    generate_detailed_metrics(model_identifier, test_set["labels"], predictions, class_labels)


# Main Script Execution
def main():
    # Load dataset
    train_set, val_set, test_set = fetch_and_prepare_emotion_data()

    # Emotion labels for GoEmotions
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
    ]

    # Models to test
    model_identifiers = ["bert-base-uncased", "gpt2"]

    # Run pipeline for each model
    for model_identifier in model_identifiers:
        try:
            execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS)
        except Exception as err:
            print(f"Error while processing {model_identifier}: {err}")


if __name__ == "__main__":
    main()
