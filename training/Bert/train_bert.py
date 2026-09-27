"""
BERT Teacher Training

This file fine-tunes a pretrained BERT-base-uncased model for
four-class text classification on the AG News dataset.

The original AG News training set is split into 90% training and
10% validation data using a fixed random seed. The original AG News
test set is kept untouched and is evaluated separately later.

After training, the best BERT model is selected based on validation
loss and saved to ./models/bert_teacher. The trained teacher is
later used to generate predictions (logits) for Knowledge
Distillation of the BiLSTM student model.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 4


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    dataset = load_dataset("fancyzhx/ag_news")

    print("\nOriginal dataset:")
    print(dataset)
    split_dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
    )

    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    # Original AG News test set remains untouched
    test_dataset = dataset["test"]

    print("\nDataset sizes:")
    print("Training:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Test:", len(test_dataset))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
    )

    print("\nTokenization complete.")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    print("\nModel loaded successfully!")
    print("Number of parameters:", model.num_parameters())

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        accuracy = (predictions == labels).mean()

        return {
            "accuracy": accuracy,
        }

    training_args = TrainingArguments(
        output_dir="./results/bert_teacher",

        num_train_epochs=2,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        learning_rate=2e-5,
        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=100,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("\nTrainer created")
    print("\nStarting BERT teacher training")

    trainer.train()

    print("\nEvaluating best model on val set")

    validation_results = trainer.evaluate(
        eval_dataset=validation_dataset
    )

    print("\nValidation results:")
    print(validation_results)
    print("\nSaving BERT teacher")

    trainer.save_model("./models/bert_teacher")
    tokenizer.save_pretrained("./models/bert_teacher")

    print("BERT teacher saved successfully")


if __name__ == "__main__":
    main()