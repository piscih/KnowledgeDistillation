"""
Select the best KD configuration using the VALIDATION split only.

Put this file in training/KD/ and run from the repo root, the same way you
run your other scripts, e.g.:

    python -m training.KD.select_on_validation

It never loads the test set. Once it names a winner, report that
configuration's existing test result (results/<name>_test_results.json)
as the headline number and label the other configurations as ablations.
"""

import json

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from training.BiLSTM.dataset import AGNewsDataset, collate_fn
from training.BiLSTM.model import BiLSTMClassifier
from training.BiLSTM.vocabulary import Vocabulary


# ============================================================
# Configuration (edit the checkpoint paths to match your files)
# ============================================================

CHECKPOINTS = {
    "T=2, alpha=0.5":  "./models/BiLSTM_KD/bilstm_kd2.pt",
    "T=4, alpha=0.5":  "./models/BiLSTM_KD/bilstm_kd.pt",
    "T=6, alpha=0.5":  "./models/BiLSTM_KD/bilstm_kd6.pt",
    "T=2, alpha=0.25": "./models/BiLSTM_KD/bilstm_kd2(0.25).pt",
    "T=2, alpha=0.75": "./models/BiLSTM_KD/bilstm_kd2(0.75).pt",
}

OUTPUT_PATH = "./results/kd_validation_selection.json"

BATCH_SIZE = 64
MIN_FREQ = 2
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print("Using device:", device)

    # Recreate the exact same train/validation split used in training
    dataset = load_dataset("fancyzhx/ag_news")
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    valid_data = split["test"]
    print(f"Validation samples: {len(valid_data)}")

    # Vocabulary must come from the training split only, as in training
    vocabulary = Vocabulary(min_freq=MIN_FREQ)
    vocabulary.build(train_data["text"])

    valid_loader = DataLoader(
        AGNewsDataset(valid_data, vocabulary),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = {}

    for name, path in CHECKPOINTS.items():
        model = BiLSTMClassifier(
            vocab_size=len(vocabulary),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["lengths"].to(device),
                )
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(batch["labels"].cpu().numpy())

        preds, labels = np.array(preds), np.array(labels)
        results[name] = {
            "checkpoint": path,
            "val_accuracy": float(accuracy_score(labels, preds)),
            "val_macro_f1": float(f1_score(labels, preds, average="macro")),
        }
        print(
            f"{name:16s} val acc {results[name]['val_accuracy'] * 100:.2f}%  "
            f"macro F1 {results[name]['val_macro_f1'] * 100:.2f}%"
        )

    winner = max(results, key=lambda k: results[k]["val_accuracy"])
    print(f"\nSelected on validation: {winner}")
    print("Report this configuration's test result as the headline number.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"selected": winner, "results": results}, f, indent=2)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()