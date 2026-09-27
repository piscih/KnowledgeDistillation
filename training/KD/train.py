import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from training.KD.dataset import AGNewsKDDataset, collate_fn
from training.BiLSTM.vocabulary import Vocabulary
from training.BiLSTM.model import BiLSTMClassifier


# ============================================================
# Configuration
# ============================================================

TRAIN_LOGITS_PATH = "./models/BERT_teacher_logits_train.pt"
VALID_LOGITS_PATH = "./models/BERT_teacher_logits_valid.pt"

MODEL_NAME = "google-bert/bert-base-uncased"

STUDENT_PATH = "./models/BiLSTM_KD/bilstm_kd6.pt"

MAX_LENGTH = 128

# Same settings as baseline BiLSTM
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

# Knowledge Distillation hyperparameters
TEMPERATURE = 6.0
ALPHA = 0.5


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# Load cached teacher logits
# ============================================================

print("\nLoading cached teacher logits...")

train_teacher_logits = torch.load(
    TRAIN_LOGITS_PATH,
    map_location="cpu",
)

valid_teacher_logits = torch.load(
    VALID_LOGITS_PATH,
    map_location="cpu",
)

print(
    f"Training teacher logits:   {train_teacher_logits.shape}"
)

print(
    f"Validation teacher logits: {valid_teacher_logits.shape}"
)


# ============================================================
# Load AG News dataset
# ============================================================

dataset = load_dataset("fancyzhx/ag_news")

train_valid = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42,
)

train_data = train_valid["train"]
valid_data = train_valid["test"]

print(f"\nTraining samples:   {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")


# ============================================================
# Build vocabulary
# ============================================================

vocabulary = Vocabulary(min_freq=MIN_FREQ)

vocabulary.build(train_data["text"])

print(f"Vocabulary size: {len(vocabulary)}")


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Create datasets
# ============================================================

train_dataset = AGNewsKDDataset(
    train_data,
    vocabulary,
    tokenizer,
    max_length=MAX_LENGTH,
)

valid_dataset = AGNewsKDDataset(
    valid_data,
    vocabulary,
    tokenizer,
    max_length=MAX_LENGTH,
)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# Create NEW BiLSTM student
# ============================================================

print("\nCreating BiLSTM student...")

student = BiLSTMClassifier(
    vocab_size=len(vocabulary),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
)

student.to(DEVICE)

print("BiLSTM student created.")


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    student.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Loss functions
# ============================================================

hard_loss_fn = nn.CrossEntropyLoss()

kd_loss_fn = nn.KLDivLoss(
    reduction="batchmean"
)


# ============================================================
# Validation
# ============================================================

def evaluate(student, data_loader, teacher_logits):

    student.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in data_loader:

            # ------------------------------------------------
            # Student inputs
            # ------------------------------------------------

            student_input_ids = batch[
                "student_input_ids"
            ].to(DEVICE)

            student_lengths = batch[
                "student_lengths"
            ]

            labels = batch[
                "labels"
            ].to(DEVICE)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            student_logits = student(
                input_ids=student_input_ids,
                lengths=student_lengths,
            )

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = hard_loss_fn(
                student_logits,
                labels,
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------

            predictions = torch.argmax(
                student_logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(data_loader)

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

print("\nStarting Knowledge Distillation training...")
print("BERT is NOT running during training.")
print("Using cached teacher logits instead.\n")


for epoch in range(NUM_EPOCHS):

    student.train()

    total_loss = 0.0
    total_hard_loss = 0.0
    total_kd_loss = 0.0


    # ========================================================
    # Training batches
    # ========================================================

    for batch in train_loader:

        # ----------------------------------------------------
        # Student inputs
        # ----------------------------------------------------

        student_input_ids = batch[
            "student_input_ids"
        ].to(DEVICE)

        student_lengths = batch[
            "student_lengths"
        ]

        labels = batch[
            "labels"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Get correct cached teacher logits
        # ----------------------------------------------------

        indices = batch[
            "indices"
        ]

        teacher_logits = train_teacher_logits[
            indices
        ].to(DEVICE)


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Student forward pass
        # ----------------------------------------------------

        student_logits = student(
            input_ids=student_input_ids,
            lengths=student_lengths,
        )


        # ----------------------------------------------------
        # Hard-label loss
        # ----------------------------------------------------

        hard_loss = hard_loss_fn(
            student_logits,
            labels,
        )


        # ----------------------------------------------------
        # Teacher soft targets
        # ----------------------------------------------------

        teacher_soft_targets = torch.softmax(
            teacher_logits / TEMPERATURE,
            dim=1,
        )


        # ----------------------------------------------------
        # Student soft predictions
        # ----------------------------------------------------

        student_log_probs = torch.log_softmax(
            student_logits / TEMPERATURE,
            dim=1,
        )


        # ----------------------------------------------------
        # Knowledge Distillation loss
        # ----------------------------------------------------

        kd_loss = kd_loss_fn(
            student_log_probs,
            teacher_soft_targets,
        )

        # Temperature scaling
        kd_loss = kd_loss * (
            TEMPERATURE ** 2
        )


        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        loss = (
            ALPHA * hard_loss
            + (1 - ALPHA) * kd_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Track losses
        # ----------------------------------------------------

        total_loss += loss.item()

        total_hard_loss += hard_loss.item()

        total_kd_loss += kd_loss.item()


    # ========================================================
    # Epoch statistics
    # ========================================================

    num_batches = len(train_loader)

    average_train_loss = (
        total_loss / num_batches
    )

    average_hard_loss = (
        total_hard_loss / num_batches
    )

    average_kd_loss = (
        total_kd_loss / num_batches
    )


    # ========================================================
    # Validation
    # ========================================================

    val_loss, val_accuracy = evaluate(
        student,
        valid_loader,
        valid_teacher_logits,
    )


    # ========================================================
    # Print results
    # ========================================================

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss:      "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Hard Loss:       "
        f"{average_hard_loss:.4f}"
    )

    print(
        f"KD Loss:         "
        f"{average_kd_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Acc:  "
        f"{val_accuracy * 100:.2f}%"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs(
            os.path.dirname(STUDENT_PATH),
            exist_ok=True,
        )

        torch.save(
            student.state_dict(),
            STUDENT_PATH,
        )

        print(
            f"✓ Best model saved "
            f"(validation accuracy: "
            f"{val_accuracy * 100:.2f}%)"
        )


# ============================================================
# Finished
# ============================================================

print("\nTraining complete.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best KD student saved to: "
    f"{STUDENT_PATH}"
)