import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from bertviz import head_view
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

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
    # Ensure the tokenizer has a pad token for GPT-2
    if "gpt2" in tokenizer.name_or_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token for GPT-2
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


# Detailed Metrics
def generate_detailed_metrics(model_identifier, true_labels, predicted_labels, label_names, results_dir):
    """Generate and save detailed evaluation metrics."""
    os.makedirs(results_dir, exist_ok=True)

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    print(f"Metrics for {model_identifier}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}\n")

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


# Modified Attention Visualization Function
def visualize_attention_with_bertviz(model_ckpt, tokenizer, sentence_a, sentence_b=None, save_path="./Results/head_view.html"):
    """Visualize attention for BERT-like models using bertviz.head_view and save it to an HTML file."""
    model = AutoModel.from_pretrained(model_ckpt, output_attentions=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Ensure the tokenizer has a pad token defined (especially for GPT-like models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    # Tokenize inputs
    viz_inputs = tokenizer(
        sentence_a, 
        sentence_b, 
        return_tensors="pt", 
        truncation=True, 
        max_length=SETTINGS["max_token_length"], 
        padding=True  # Ensures proper padding is applied
    )
    viz_inputs = {key: val.to(model.device) for key, val in viz_inputs.items()}

    # Forward pass to get attention weights
    outputs = model(**viz_inputs)
    attention = outputs.attentions

    # Compute tokens and optional sentence B start index
    tokens = tokenizer.convert_ids_to_tokens(viz_inputs["input_ids"][0])
    sentence_b_start = None if sentence_b is None else (viz_inputs["token_type_ids"] == 0).sum(dim=1).item()

    # Render head view visualization
    print(f"Visualizing attention for: '{sentence_a}' {'and ' + sentence_b if sentence_b else ''}")
    try:
        html = head_view(attention, tokens, sentence_b_start, heads=[8])
        if html is None:
            raise ValueError("head_view returned None instead of HTML content.")
        # Save the visualization to an HTML file
        with open(save_path, "w") as f:
            f.write(html)
        print(f"Head view visualization saved to: {save_path}")
    except Exception as e:
        print(f"An error occurred while generating the attention visualization: {e}")


def execute_model_pipeline(model_identifier, train_set, val_set, test_set, class_labels, config):
    """Train and evaluate the specified model."""
    print(f"Initializing {model_identifier}...")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    # Handle GPT-2 padding token specifics
    if "gpt2" in model_identifier:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_identifier,
        num_labels=len(class_labels),
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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
    print(f"Evaluating {model_identifier}...")
    predictions = trainer.predict(test_set)
    results_dir = f"{output_dir}/metrics"
    generate_detailed_metrics(
        model_identifier,
        predictions.label_ids,
        predictions.predictions.argmax(-1),
        class_labels,
        results_dir,
    )

    # Attention Visualization (only for BERT)
    if "bert" in model_identifier:
        example_text_a = "The movie was fantastic and full of surprises."
        example_text_b = "I enjoyed the film thoroughly."
        visualize_attention_with_bertviz(model_identifier, tokenizer, example_text_a, example_text_b)


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
