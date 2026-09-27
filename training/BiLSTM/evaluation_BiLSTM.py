"""
BiLSTM Baseline Evaluation

This file evaluates the trained baseline BiLSTM classifier on the
untouched AG News test set.

The same 90/10 training split and random seed used during training
are recreated so that the vocabulary can be rebuilt consistently.
The vocabulary is constructed only from the training data to avoid
data leakage.

The saved BiLSTM model is loaded and evaluated using accuracy,
macro-averaged precision, recall, F1 score, and per-class metrics.
A confusion matrix is also generated to visualize the classification
performance across the four AG News categories.

The evaluation results are saved as a JSON file and the confusion
matrix is saved as an image for later comparison with the BERT
teacher and Knowledge Distillation models.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay 
)
import json
from vocabulary import Vocabulary
from dataset import AGNewsDataset, collate_fn
from model import BiLSTMClassifier
import matplotlib.pyplot as plt 



BATCH_SIZE = 64
MIN_FREQ = 2

MODEL_PATH = "./models/BiLSTM/bilstm_student.pt"
RESULTS_PATH = "./results/bilstm_baseline_results.json"

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")

def main():

    device = get_device()

    print("Using device:", device)
    print("\nLoading AG News dataset")

    dataset = load_dataset("fancyzhx/ag_news")
    split_dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42
    )

    train_data = split_dataset["train"]
    test_data = dataset["test"]

    print("\nDataset sizes:")
    print("Training:", len(train_data))
    print("Test:", len(test_data))

    print("\nBuilding vocabulary")

    vocabulary = Vocabulary(
        min_freq=MIN_FREQ
    )
    vocabulary.build(
        train_data["text"]
    )

    print(
        "Vocabulary size:",
        len(vocabulary)
    )

    test_dataset = AGNewsDataset(
        test_data,
        vocabulary
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BiLSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )

    print("\nLoading trained model")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel information:")
    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    all_predictions = []
    all_labels = []

    print("\nRunning evaluation")

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)

            lengths = batch["lengths"]

            labels = batch["labels"].to(device)

            logits = model(
                input_ids,
                lengths
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=[0, 1, 2, 3],
            zero_division=0
        )
    )

    class_names = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]

    per_class = {}

    for i, class_name in enumerate(class_names):

        per_class[class_name] = {
            "precision": float(class_precision[i]),
            "recall": float(class_recall[i]),
            "f1": float(class_f1[i]),
        }
    results = {

        "model": MODEL_PATH,

        "dataset": "fancyzhx/ag_news",

        "test_samples": len(all_labels),

        "accuracy": float(accuracy),

        "macro_precision": float(macro_precision),

        "macro_recall": float(macro_recall),

        "macro_f1": float(macro_f1),

        "parameters": total_parameters,

        "per_class": per_class,
    }

    import os

    os.makedirs(
        "./results",
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print("\n")
    print("=" * 60)
    print("BiLSTM TEST RESULTS")
    print("=" * 60)

    print(
        json.dumps(
            results,
            indent=2
        )
    )

    print("=" * 60)

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )
    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nConfusion Matrix:")
    print(matrix)

    class_names = ["World", "Sports", "Business", "Sci/Tech"] 
    disp = ConfusionMatrixDisplay( confusion_matrix=matrix, display_labels=class_names ) 
    disp.plot(cmap="Blues", values_format="d")
    plt.title("BiLSTM Confusion Matrix") 
    plt.tight_layout() 
    plt.savefig("./results/bilstm_confusion_matrix.png", dpi=300) 
    plt.close() 
    print("Confusion matrix saved to: ./results/bilstm_confusion_matrix.png")


if __name__ == "__main__":
    main()