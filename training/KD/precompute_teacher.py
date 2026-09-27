import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from training.KD.dataset import AGNewsKDDataset, collate_fn
from training.BiLSTM.vocabulary import Vocabulary


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google-bert/bert-base-uncased"
TEACHER_PATH = "./models/bert_teacher"

TRAIN_LOGITS_PATH = "./models/BERT_teacher_logits_train.pt"
VALID_LOGITS_PATH = "./models/BERT_teacher_logits_valid.pt"

MAX_LENGTH = 128
BATCH_SIZE = 64
MIN_FREQ = 2


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
# Load dataset
# ============================================================

dataset = load_dataset("fancyzhx/ag_news")

train_valid = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

train_data = train_valid["train"]
valid_data = train_valid["test"]


# ============================================================
# Build vocabulary
# ============================================================

vocabulary = Vocabulary(min_freq=MIN_FREQ)
vocabulary.build(train_data["text"])


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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
    shuffle=False,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# Load BERT teacher
# ============================================================

print("\nLoading BERT teacher...")

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_PATH
)

teacher.to(DEVICE)
teacher.eval()

for parameter in teacher.parameters():
    parameter.requires_grad = False

print("Teacher loaded.")


# ============================================================
# Function to precompute logits
# ============================================================

def precompute_logits(data_loader, save_path):

    all_logits = []

    with torch.no_grad():

        for batch_number, batch in enumerate(data_loader):

            teacher_input_ids = batch[
                "teacher_input_ids"
            ].to(DEVICE)

            teacher_attention_mask = batch[
                "teacher_attention_mask"
            ].to(DEVICE)

            outputs = teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )

            logits = outputs.logits.cpu()

            all_logits.append(logits)

            if (batch_number + 1) % 100 == 0:
                print(
                    f"Processed "
                    f"{batch_number + 1}/{len(data_loader)} batches"
                )

    all_logits = torch.cat(all_logits, dim=0)

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    torch.save(
        all_logits,
        save_path,
    )

    print(
        f"\nSaved teacher logits:"
        f"\n{save_path}"
        f"\nShape: {all_logits.shape}"
    )


# ============================================================
# Precompute training logits
# ============================================================

print("\nPrecomputing training teacher logits...")

precompute_logits(
    train_loader,
    TRAIN_LOGITS_PATH,
)


# ============================================================
# Precompute validation logits
# ============================================================

print("\nPrecomputing validation teacher logits...")

precompute_logits(
    valid_loader,
    VALID_LOGITS_PATH,
)


print("\nTeacher-logit precomputation complete.")