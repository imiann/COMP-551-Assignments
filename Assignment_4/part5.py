import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
from bertviz.head_view import show
from IPython.display import display

# Centralized Configuration
SETTINGS = {
    "max_token_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 5e-5,
    "epochs": 3,
    "weight_decay_factor": 0.01,
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
        split = split.map(lambda x: {"labels": x["labels"][0]}, batched=False)

    return train_set, val_set, test_set


# Metric Computation
def evaluate_predictions(predictions):
    """Calculate metrics for evaluation."""
    true_labels = predictions.label_ids
    predicted_labels = predictions.predictions.argmax(-1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# Visualization
def visualize_attention_with_bertviz(model, tokenizer, input_text):
    """Visualize attention matrices using bertviz."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=SETTINGS["max_token_length"])
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move inputs to device
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    show(model, tokenizer, inputs["input_ids"], attentions)


# Model Pipeline
def execute_model_pipeline(model_identifier, train_set, val_set, test_set, config):
    """Train and evaluate the specified model."""
    print(f"Initializing {model_identifier}...")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_identifier,
        num_labels=28,  # Number of emotion labels in GoEmotions
        output_attentions=True  # Enable attention outputs
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Tokenize datasets
    def tokenize_data(dataset):
        return dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=config["max_token_length"],
            ),
            batched=True,
        )

    train_set = tokenize_data(train_set)
    val_set = tokenize_data(val_set)
    test_set = tokenize_data(test_set)

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training configuration
    training_args = TrainingArguments(
        output_dir=f"{config['output_directory']}/{model_identifier.split('/')[-1]}",
        evaluation_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay_factor"],
        logging_dir=f"{config['log_directory']}/{model_identifier.split('/')[-1]}",
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
    evaluation_results = trainer.evaluate(test_set)
    print(f"Evaluation Results for {model_identifier}: {evaluation_results}")

    # Visualize Attention for Examples
    example_text_correct = "The movie was fantastic and full of surprises."
    example_text_incorrect = "The product failed miserably."

    print("Visualizing Attention for Correct Example:")
    visualize_attention_with_bertviz(model, tokenizer, example_text_correct)

    print("Visualizing Attention for Incorrect Example:")
    visualize_attention_with_bertviz(model, tokenizer, example_text_incorrect)


# Main Script Execution
def main():
    # Load dataset
    train_set, val_set, test_set = fetch_and_prepare_emotion_data()

    # Models to test
    model_identifiers = ["bert-base-uncased", "gpt2"]

    for model_identifier in model_identifiers:
        execute_model_pipeline(model_identifier, train_set, val_set, test_set, SETTINGS)


if __name__ == "__main__":
    main()
