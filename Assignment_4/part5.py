import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForPreTraining,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
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

def pretrain_with_mlm(model_name, train_set, tokenizer):
    """Perform masked language modeling (MLM) on the train set."""
    print(f"Pre-training {model_name} with MLM...")

    # Load model configuration and model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForPreTraining.from_pretrained(model_name, config=config)

    # Tokenize data and add `labels`
    def add_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=SETTINGS["max_token_length"],
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Copy input_ids as labels
        return tokenized_inputs

    tokenized_data = train_set.map(add_labels, batched=True)
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Adjust model configuration
    config.num_labels = config.vocab_size  # Align with vocab size
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match tokenizer

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{SETTINGS['output_directory']}/{model_name.split('/')[-1]}_mlm",
        evaluation_strategy="epoch",
        learning_rate=SETTINGS["learning_rate"],
        per_device_train_batch_size=SETTINGS["train_batch_size"],
        num_train_epochs=1,  # Short pre-training
        save_total_limit=1,
        logging_dir=f"{SETTINGS['log_directory']}/{model_name.split('/')[-1]}_mlm",
        logging_steps=SETTINGS["log_interval_steps"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    print(f"Finished pre-training {model_name} with MLM.")
    return model





from transformers import DataCollatorForLanguageModeling

def pretrain_with_nsp(model_name, train_set, tokenizer):
    """Perform next sentence prediction (NSP) on the train set."""
    print(f"Pre-training {model_name} with NSP...")

    model = AutoModelForPreTraining.from_pretrained(model_name)

    # Prepare NSP-specific dataset
    def create_nsp_data(examples):
        sentences = examples["text"]
        n = len(sentences)
        pairs = []
        labels = []

        for i in range(0, n - 1, 2):  # Pair sentences
            if i + 1 < n:
                # Pair consecutive sentences as positive examples
                pairs.append((sentences[i], sentences[i + 1]))
                labels.append(1)

                # Pair non-consecutive sentences as negative examples
                random_idx = (i + 2) % n
                pairs.append((sentences[i], sentences[random_idx]))
                labels.append(0)

        first_sentences, second_sentences = zip(*pairs)
        return {"sentence1": first_sentences, "sentence2": second_sentences, "labels": labels}

    # Map dataset to NSP pairs
    nsp_dataset = train_set.map(create_nsp_data, batched=True, remove_columns=train_set.column_names)
    tokenized_data = tokenizer(
        list(nsp_dataset["sentence1"]),
        text_pair=list(nsp_dataset["sentence2"]),
        truncation=True,
        padding="max_length",
        max_length=SETTINGS["max_token_length"],
        return_tensors="pt",
    )
    tokenized_data["labels"] = torch.tensor(nsp_dataset["labels"])  # Add labels for NSP

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # MLM is not applied for NSP
    )

    # Wrap into a PyTorch Dataset
    dataset = torch.utils.data.TensorDataset(
        tokenized_data["input_ids"],
        tokenized_data["attention_mask"],
        tokenized_data["token_type_ids"],
        tokenized_data["labels"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{SETTINGS['output_directory']}/{model_name.split('/')[-1]}_nsp",
        evaluation_strategy="epoch",
        learning_rate=SETTINGS["learning_rate"],
        per_device_train_batch_size=SETTINGS["train_batch_size"],
        num_train_epochs=1,  # Short pre-training
        save_total_limit=1,
        logging_dir=f"{SETTINGS['log_directory']}/{model_name.split('/')[-1]}_nsp",
        logging_steps=SETTINGS["log_interval_steps"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"Finished pre-training {model_name} with NSP.")
    return model



def execute_model_pipeline(model_identifier, train_set, val_set, test_set, class_labels, config, pretrained_model=None):
    """Train and evaluate the specified model."""
    print(f"Initializing {model_identifier}...")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    # Handle missing padding tokens for GPT-2
    if "gpt2" in model_identifier:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        tokenizer.padding_side = "left"  # Ensure left-padding for causal models

    if pretrained_model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_identifier,
            num_labels=len(class_labels),
        )
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
    print(f"Evaluating {model_identifier}...")
    evaluation_results = trainer.evaluate(test_set)
    print(f"Evaluation Results for {model_identifier}: {evaluation_results}")

    # Generate metrics and save them
    predictions = trainer.predict(test_set).predictions.argmax(-1)
    generate_detailed_metrics(model_identifier, test_set["labels"], predictions, class_labels, output_dir)


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
        # print(f"Running without pretraining: {model_identifier}")
        # execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS)

        print(f"Running with MLM pretraining: {model_identifier}")
        mlm_model = pretrain_with_mlm(model_identifier, train_set, AutoTokenizer.from_pretrained(model_identifier))
        execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS, pretrained_model=mlm_model)

        # print(f"Running with NSP pretraining: {model_identifier}")
        # nsp_model = pretrain_with_nsp(model_identifier, train_set, AutoTokenizer.from_pretrained(model_identifier))
        # execute_model_pipeline(model_identifier, train_set, val_set, test_set, emotion_labels, SETTINGS, pretrained_model=nsp_model)


if __name__ == "__main__":
    main()
