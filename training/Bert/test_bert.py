"""
BERT Teacher Evaluation

This file evaluates the fine-tuned BERT teacher on the untouched
AG News test set.

The trained teacher and its tokenizer are loaded from
./models/bert_teacher. Predictions are generated for all test
examples, and the evaluation includes accuracy, macro-averaged
precision, recall, F1 score, and per-class metrics for the four
AG News categories.

The resulting metrics are saved as a JSON file and are later used
to compare the BERT teacher with the BiLSTM baseline and the
Knowledge Distillation models.
"""

import json
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)


MODEL_PATH = "./models/bert_teacher"
MAX_LENGTH = 128


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("\nLoading fine-tuned BERT model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)

    print("Model loaded successfully!")
    print("Number of parameters:", model.num_parameters())
    print("\nLoading AG News test dataset")

    dataset = load_dataset("fancyzhx/ag_news")

    test_dataset = dataset["test"]

    print("Test examples:", len(test_dataset))
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    print("\nRunning predictions on test set...")

    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    labels = predictions.label_ids

    predicted_labels = np.argmax(logits, axis=-1)
    accuracy = np.mean(predicted_labels == labels)

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    class_names = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]

    metrics = {}

    for class_id, class_name in enumerate(class_names):

        true_positive = np.sum(
            (predicted_labels == class_id) &
            (labels == class_id)
        )

        false_positive = np.sum(
            (predicted_labels == class_id) &
            (labels != class_id)
        )

        false_negative = np.sum(
            (predicted_labels != class_id) &
            (labels == class_id)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")

    macro_precision = np.mean([
        metrics[name]["precision"]
        for name in class_names
    ])

    macro_recall = np.mean([
        metrics[name]["recall"]
        for name in class_names
    ])

    macro_f1 = np.mean([
        metrics[name]["f1"]
        for name in class_names
    ])

    print("\n" + "-" * 50)
    print("MACRO AVERAGES")
    print("-" * 50)

    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"F1:        {macro_f1:.4f}")

    results = {
        "model": MODEL_PATH,
        "dataset": "fancyzhx/ag_news",
        "test_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": float(metrics[name]["precision"]),
                "recall": float(metrics[name]["recall"]),
                "f1": float(metrics[name]["f1"]),
            }
            for name in class_names
        },
    }

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:")
    print("./results/bert_test_results.json")


if __name__ == "__main__":
    main()