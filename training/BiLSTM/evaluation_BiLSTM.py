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


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64
MIN_FREQ = 2

MODEL_PATH = "./models/BiLSTM/bilstm_student.pt"
RESULTS_PATH = "./results/bilstm_baseline_results.json"

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    device = get_device()

    print("Using device:", device)

    # --------------------------------------------------------
    # 1. Load AG News dataset
    # --------------------------------------------------------

    print("\nLoading AG News dataset...")

    dataset = load_dataset("fancyzhx/ag_news")

    # Same split used during training
    split_dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42
    )

    train_data = split_dataset["train"]
    test_data = dataset["test"]

    print("\nDataset sizes:")
    print("Training:", len(train_data))
    print("Test:", len(test_data))

    # --------------------------------------------------------
    # 2. Rebuild vocabulary
    # --------------------------------------------------------

    print("\nBuilding vocabulary...")

    vocabulary = Vocabulary(
        min_freq=MIN_FREQ
    )

    # IMPORTANT:
    # Vocabulary must be built only from training data.
    vocabulary.build(
        train_data["text"]
    )

    print(
        "Vocabulary size:",
        len(vocabulary)
    )

    # --------------------------------------------------------
    # 3. Create test dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. Create model
    # --------------------------------------------------------

    model = BiLSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )

    # --------------------------------------------------------
    # 5. Load trained model
    # --------------------------------------------------------

    print("\nLoading trained model...")

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

    # --------------------------------------------------------
    # 6. Count parameters
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7. Run inference
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    print("\nRunning evaluation...")

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


    # --------------------------------------------------------
    # 10. Calculate metrics
    # --------------------------------------------------------

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

    # Per-class metrics
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

    # --------------------------------------------------------
    # 11. Create results dictionary
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 12. Save JSON
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 13. Print results
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 14. Confusion matrix
    # --------------------------------------------------------

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

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()