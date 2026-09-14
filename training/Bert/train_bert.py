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

    # --------------------------------------------------
    # 1. Check device
    # --------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # --------------------------------------------------
    # 2. Load AG News
    # --------------------------------------------------
    dataset = load_dataset("fancyzhx/ag_news")

    print("\nOriginal dataset:")
    print(dataset)

    # --------------------------------------------------
    # 3. Split original training data
    #    into train + validation
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 4. Load tokenizer
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    # --------------------------------------------------
    # 5. Tokenize datasets
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 6. Data collator
    # --------------------------------------------------
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------
    # 7. Load BERT
    # --------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    print("\nModel loaded successfully!")
    print("Number of parameters:", model.num_parameters())

    # --------------------------------------------------
    # 8. Metrics
    # --------------------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        accuracy = (predictions == labels).mean()

        return {
            "accuracy": accuracy,
        }

    # --------------------------------------------------
    # 9. Training arguments
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./results/bert_teacher",

        num_train_epochs=2,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        learning_rate=2e-5,
        weight_decay=0.01,

        # Evaluate on validation set
        eval_strategy="epoch",

        # Save checkpoint after every epoch
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=100,

        # Select best model based on validation loss
        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
    )

    # --------------------------------------------------
    # 10. Create Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("\nTrainer created successfully!")

    # --------------------------------------------------
    # 11. Train
    # --------------------------------------------------
    print("\nStarting BERT teacher training...")

    trainer.train()

    # --------------------------------------------------
    # 12. Evaluate on validation set
    # --------------------------------------------------
    print("\nEvaluating best model on validation set...")

    validation_results = trainer.evaluate(
        eval_dataset=validation_dataset
    )

    print("\nValidation results:")
    print(validation_results)

    # --------------------------------------------------
    # 13. Save final best model
    # --------------------------------------------------
    print("\nSaving BERT teacher...")

    trainer.save_model("./models/bert_teacher")
    tokenizer.save_pretrained("./models/bert_teacher")

    print("BERT teacher saved successfully!")

    # --------------------------------------------------
    # IMPORTANT:
    # The test set is NOT evaluated here.
    #
    # We will evaluate it later using test.py.
    # --------------------------------------------------


if __name__ == "__main__":
    main()