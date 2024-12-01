import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
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


# Compute class weights for imbalanced datasets
def compute_class_weights(labels):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return torch.tensor(class_weights, dtype=torch.float)


# Define evaluation metrics
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, f1_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# Train and evaluate the model
def train_with_trainer(model_name):
    # Load and preprocess data
    train_data, val_data, test_data = load_data()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Handle models like GPT-2 that don't have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=28)

    # Tokenize datasets
    train_data = tokenize_data(train_data, tokenizer)
    val_data = tokenize_data(val_data, tokenizer)
    test_data = tokenize_data(test_data, tokenizer)

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Compute class weights
    labels = np.array(train_data["labels"])
    class_weights = compute_class_weights(labels).to("cuda" if torch.cuda.is_available() else "cpu")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.split('/')[-1]}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs_{model_name.split('/')[-1]}",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=None,  # Default data collator works
    )

    # Train the model
    print(f"Training the model: {model_name}")
    trainer.train()

    # Evaluate the model
    print(f"Evaluating the model: {model_name}")
    test_results = trainer.evaluate(test_data)
    print(f"Test results for {model_name}: {test_results}")

    return test_results


# Main function
def main():
    # Define models to evaluate
    models = [
        "bert-base-uncased",
        "gpt2",
    ]

    # Train and evaluate each model
    for model_name in models:
        try:
            results = train_with_trainer(model_name)
            print(f"Test results for {model_name}: {results}")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")


if __name__ == "__main__":
    main()
