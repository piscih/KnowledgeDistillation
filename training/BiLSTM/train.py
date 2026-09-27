"""
BiLSTM Baseline Training

This file trains the baseline BiLSTM classifier for the AG News
text classification task.

The original AG News training set is split into 90% training and
10% validation data using the same fixed seed as the BERT teacher.
The vocabulary is built only from the training data to avoid
information leakage.

The BiLSTM is trained using standard cross-entropy loss with the
ground-truth labels. No teacher predictions or Knowledge Distillation
loss are used in this baseline.

The model with the highest validation accuracy is saved and is later
evaluated on the untouched AG News test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from vocabulary import Vocabulary
from dataset import AGNewsDataset, collate_fn
from model import BiLSTMClassifier

BATCH_SIZE = 64
NUM_EPOCHS = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

MIN_FREQ = 2

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3

MODEL_PATH = "./models/BiLSTM/bilstm.pt"

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")


def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]
            labels = batch["labels"].to(device)

            logits = model(
                input_ids,
                lengths
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(
            input_ids,
            lengths
        )
        loss = criterion(
            logits,
            labels
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy

def main():
    device = get_device()

    print("Using device:", device)
    dataset = load_dataset(
        "fancyzhx/ag_news"
    )

    print("\nOriginal dataset:")
    print(dataset)
    split_dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
    )

    train_data = split_dataset["train"]
    validation_data = split_dataset["test"]
    test_data = dataset["test"]

    print("\nDataset sizes:")
    print("Training:", len(train_data))
    print("Validation:", len(validation_data))
    print("Test:", len(test_data))
    print("\nBuilding vocabulary...")
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

    train_dataset = AGNewsDataset(
        train_data,
        vocabulary
    )

    validation_dataset = AGNewsDataset(
        validation_data,
        vocabulary
    )

    test_dataset = AGNewsDataset(
        test_data,
        vocabulary
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BiLSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )

    model = model.to(device)

    print("\nBiLSTM model created!")

    print(
        "Number of parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_accuracy = 0.0

    print("\nStarting BiLSTM training\n")

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print("-" * 60)
        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(vocabulary),
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "dropout": DROPOUT,
                    "vocabulary": vocabulary,
                },
                MODEL_PATH,
            )

            print(
                "Best model saved!"
            )

    print("\nTraining complete!")

    print(
        "Best validation accuracy:",
        best_validation_accuracy
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )

if __name__ == "__main__":
    main()