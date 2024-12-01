import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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
    "output_directory": "./Results",
    "log_directory": "./Results/logs",
}

# Ensure the Results directory exists
os.makedirs(SETTINGS["output_directory"], exist_ok=True)
os.makedirs(SETTINGS["log_directory"], exist_ok=True)


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
    if "gpt2" in tokenizer.name_or_path:
        # Ensure the padding token is set for GPT-2
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.padding_side = "left"

    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",  # Consistent padding across batches
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )


# Metric Computation
def evaluate_predictions(predictions):
    """Calculate metrics for evaluation."""
    true_labels = predictions.label_ids
    predicted_labels = predictions.predictions.argmax(-1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# Manual Attention Visualization
def visualize_attention_manual(attention, tokens, output_dir, model_identifier):
    """
    Manual visualization for attention using Matplotlib. Saves plots in a dedicated folder for each model.
    """
    model_output_dir = os.path.join(output_dir, model_identifier)
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        for layer in range(len(attention)):  # Iterate through layers
            for head in range(attention[layer].shape[1]):  # Iterate through heads
                attention_matrix = attention[layer][0][head].detach().cpu().numpy()

                print(f"Visualizing attention for Layer {layer}, Head {head}...")
                plt.figure(figsize=(10, 8))
                plt.imshow(attention_matrix, cmap='viridis')
                plt.colorbar()
                plt.title(f"Attention Head {head} - Layer {layer}")
                plt.xlabel("Tokens Attended To")
                plt.ylabel("Tokens Attending")
                plt.xticks(range(len(tokens)), tokens, rotation=90)
                plt.yticks(range(len(tokens)), tokens)
                plt.tight_layout()

                save_path = os.path.join(model_output_dir, f"attention_layer{layer}_head{head}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved attention visualization to {save_path}")

    except Exception as e:
        print(f"An error occurred during manual attention visualization: {e}")


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


# Confusion Matrix Plotting
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


def execute_model_pipeline(model_identifier, train_set, val_set, test_set, class_labels, config, phase="untrained"):
    """Evaluate or train and evaluate the specified model."""
    print(f"Initializing {model_identifier} ({phase})...")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    # Handle missing padding tokens for GPT-2
    if "gpt2" in model_identifier:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_identifier,
        num_labels=len(class_labels),
    )

    # Adjust model configuration for GPT-2
    if "gpt2" in model_identifier:
        print("Configuring GPT-2...")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings for new tokens

    # Tokenize datasets
    train_set = tokenize_data(train_set, tokenizer, config["max_token_length"])
    val_set = tokenize_data(val_set, tokenizer, config["max_token_length"])
    test_set = tokenize_data(test_set, tokenizer, config["max_token_length"])

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "text"])
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "text"])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "text"])

    # Results folder
    output_dir = f"{config['output_directory']}/{phase}/{model_identifier.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    if phase == "trained":
        # Training configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["train_batch_size"],
            per_device_eval_batch_size=config["eval_batch_size"],
            num_train_epochs=config["epochs"],
            weight_decay=config["weight_decay_factor"],
            logging_dir=f"{config['log_directory']}/{model_identifier.split('/')[-1]}",
            logging_steps=config["log_interval_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

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

        # Evaluate using Trainer
        predictions = trainer.predict(test_set)

    else:
        # Evaluate without training
        predictions = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=config["eval_batch_size"],
            ),
            eval_dataset=test_set,
            tokenizer=tokenizer,
            compute_metrics=evaluate_predictions,
        ).predict(test_set)

    # Save metrics and plots
    generate_detailed_metrics(
        model_identifier.split("/")[-1],
        predictions.label_ids,
        predictions.predictions.argmax(-1),
        class_labels,
        output_dir,
    )

    # Manual Visualization
    print(f"Generating manual attention visualizations ({phase})...")
    model.config.output_attentions = True
    for idx in range(2):  # First 2 examples
        example_text = test_set["text"][idx]
        inputs = tokenizer(
            example_text,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_token_length"],
            padding=True
        ).to(model.device)

        outputs = model(**inputs)
        attention = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        visualize_attention_manual(attention, tokens, output_dir, model_identifier.split("/")[-1])


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
    model_identifiers = ["gpt2", "bert-base-uncased", "distilbert-base-uncased", "roberta-base"]

    for model_identifier in model_identifiers:
        # Evaluate pre-trained (untrained) model
        execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS, phase="untrained")

        # Train and re-evaluate the model
        execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS, phase="trained")


if __name__ == "__main__":
    main()
