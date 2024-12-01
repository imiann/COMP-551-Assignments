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
from bertviz.head_view import head_view
from IPython.display import display, HTML

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

    train_set = dataset["train"].filter(single_label_filter).select(range(3200))
    val_set = dataset["validation"].filter(single_label_filter).select(range(3200))
    test_set = dataset["test"].filter(single_label_filter).select(range(3200))

    # Simplify label structure
    for split in [train_set, val_set, test_set]:
        split.map(lambda x: {"labels": x["labels"][0]}, batched=False)

    return train_set, val_set, test_set


# Tokenization Handler
def tokenize_data(dataset, tokenizer, max_length):
    """Apply tokenization to text data."""
    if "gpt2" in tokenizer.name_or_path:  # Handle GPT-2 specifics
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        tokenizer.padding_side = "left"  # GPT-2 requires left-padding for causal models

    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
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


# Visualization and Metrics
def generate_detailed_metrics(model_identifier, true_labels, predicted_labels, label_names, results_dir):
    """Generate and save detailed evaluation metrics."""
    os.makedirs(results_dir, exist_ok=True)

    print(f"Metrics for {model_identifier}:")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"Weighted F1-Score: {f1_score(true_labels, predicted_labels, average='weighted'):.4f}\n")

    cmatrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", cmatrix)

    report = classification_report(true_labels, predicted_labels, target_names=label_names, digits=4)
    print("\nClassification Report:\n", report)

    # Save metrics to files
    with open(f"{results_dir}/classification_report.txt", "w") as report_file:
        report_file.write(report)

    np.savetxt(f"{results_dir}/confusion_matrix.csv", cmatrix, delimiter=",", fmt="%d")

    create_confusion_matrix_plot(cmatrix, label_names, model_identifier, results_dir)


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
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.close()


# Add attention visualization with bertviz
def visualize_attention_with_bertviz(model, tokenizer, input_text, layer_index=0, head_index=0):
    """Visualize attention for a single transformer block and attention head."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=SETTINGS["max_token_length"])
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move inputs to device

    # Perform inference and collect attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Extract attention manually from hidden states (if model allows this)
    attention = outputs.hidden_states[layer_index]  # Replace with your specific logic if available
    attentions = attention[:, head_index].detach().cpu().numpy()  # Adjust for head

    # Tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Display the attention matrix for the specified layer and head
    print(f"Visualizing Attention - Layer {layer_index}, Head {head_index}")
    plt.figure(figsize=(12, 8))
    plt.imshow(attentions, interpolation="nearest", cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Attention Matrix - Layer {layer_index}, Head {head_index}")
    plt.xlabel("Tokens Attended To")
    plt.ylabel("Tokens Attending")
    plt.tight_layout()
    plt.savefig(f"./Results/attention_{layer_index}_{head_index}.png")



def execute_model_pipeline(model_identifier, train_set, val_set, test_set, class_labels, config, pretrained_model=None):
    """Train and evaluate the specified model."""
    print(f"Initializing {model_identifier}...")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    # Handle GPT-2 specifics
    if "gpt2" in model_identifier:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    if pretrained_model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_identifier,
            num_labels=len(class_labels),
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        model = pretrained_model
        model.num_labels = len(class_labels)

    # Tokenize datasets
    train_set = tokenize_data(train_set, tokenizer, config["max_token_length"])
    val_set = tokenize_data(val_set, tokenizer, config["max_token_length"])
    test_set = tokenize_data(test_set, tokenizer, config["max_token_length"])

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training configuration
    output_dir = f"{config['output_directory']}/{model_identifier.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

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

    # Evaluation
    model.config.output_attentions = True  # Enable attentions for evaluation
    if "bert" in model_identifier:  # Only visualize attention for BERT-like models
        example_text = "The movie was fantastic and full of surprises."
        print(f"Visualizing attention for {model_identifier}:")
        visualize_attention_with_bertviz(model, tokenizer, example_text, layer_index=0, head_index=0)


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

    for model_identifier in model_identifiers:
        execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS)


if __name__ == "__main__":
    main()
