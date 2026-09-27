import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report

from training.BiLSTM.vocabulary import Vocabulary
from training.BiLSTM.model import BiLSTMClassifier
from training.BiLSTM.dataset import AGNewsDataset, collate_fn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay 
)

import matplotlib.pyplot as plt 


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./models/BiLSTM_KD/bilstm_kd.pt"

BATCH_SIZE = 64
MIN_FREQ = 2

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Check device
    # --------------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # --------------------------------------------------------
    # 2. Load AG News
    # --------------------------------------------------------

    print("\nLoading AG News...")

    dataset = load_dataset("fancyzhx/ag_news")

    # Original AG News training set
    full_train_dataset = dataset["train"]

    # Original AG News test set
    test_dataset = dataset["test"]

    print("Original training samples:", len(full_train_dataset))
    print("Test samples:", len(test_dataset))

    # --------------------------------------------------------
    # 3. Recreate EXACT training/validation split
    #    used during BiLSTM/KD training
    # --------------------------------------------------------

    split = full_train_dataset.train_test_split(
        test_size=0.1,
        seed=42,
    )

    train_dataset = split["train"]
    valid_dataset = split["test"]

    print("\nRecreated training split:")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(valid_dataset))
    print("Test samples:", len(test_dataset))

    # --------------------------------------------------------
    # 4. Build vocabulary from SAME training data
    # --------------------------------------------------------

    print("\nBuilding vocabulary...")

    vocabulary = Vocabulary(
        min_freq=MIN_FREQ
    )

    vocabulary.build(
        train_dataset["text"]
    )

    print("Vocabulary size:", len(vocabulary))

    # --------------------------------------------------------
    # 5. Prepare test dataset
    # --------------------------------------------------------

    test_dataset = AGNewsDataset(
        test_dataset,
        vocabulary,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # 6. Create BiLSTM model
    # --------------------------------------------------------

    print("\nCreating KD BiLSTM...")

    model = BiLSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    # --------------------------------------------------------
    # 7. Load trained KD model
    # --------------------------------------------------------

    print("Loading model from:", MODEL_PATH)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    print("Model loaded successfully!")

    # --------------------------------------------------------
    # 8. Evaluation
    # --------------------------------------------------------

    model.eval()

    all_predictions = []
    all_labels = []

    print("\nRunning predictions on test set...")

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids,
                lengths,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Convert to NumPy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # --------------------------------------------------------
    # 9. Overall metrics
    # --------------------------------------------------------

    accuracy = np.mean(
        all_predictions == all_labels
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    # --------------------------------------------------------
    # 10. Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL KD BiLSTM TEST RESULTS")
    print("=" * 60)

    print("Test samples:", len(all_labels))

    print(
        f"Accuracy        : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision : {macro_precision:.4f} "
        f"({macro_precision * 100:.2f}%)"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f} "
        f"({macro_recall * 100:.2f}%)"
    )

    print(
        f"Macro F1        : {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # 11. Per-class results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PER-CLASS RESULTS")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        class_metrics = report[class_name]

        print(f"\n{class_name}:")
        print(
            f"  Precision: "
            f"{class_metrics['precision']:.4f}"
        )
        print(
            f"  Recall:    "
            f"{class_metrics['recall']:.4f}"
        )
        print(
            f"  F1:        "
            f"{class_metrics['f1-score']:.4f}"
        )

    # --------------------------------------------------------
    # 12. Save results
    # --------------------------------------------------------

    results = {
        "model": MODEL_PATH,
        "dataset": "fancyzhx/ag_news",
        "test_samples": len(all_labels),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            class_name: {
                "precision": float(
                    report[class_name]["precision"]
                ),
                "recall": float(
                    report[class_name]["recall"]
                ),
                "f1": float(
                    report[class_name]["f1-score"]
                ),
            }
            for class_name in CLASS_NAMES
        },
    }

    output_path = "./results/bilstm_kd4_test_results.json"

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("Results saved to:")
    print(output_path)
    print("=" * 60)

    print("\nTesting complete!")


        # --------------------------------------------------------
    # 13. Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title(
        "BiLSTM + Knowledge Distillation Confusion Matrix"
    )

    plt.tight_layout()

    confusion_matrix_path = (
        "./results/bilstm_kd4_confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
    )

    plt.close()

    print(
        "Confusion matrix saved to:",
        confusion_matrix_path,
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()