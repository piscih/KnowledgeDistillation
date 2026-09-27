"""
BERT Teacher Model

This file defines the BERT teacher model used for knowledge distillation
on the AG News text classification task.

The teacher is based on the pretrained BERT-base-uncased model and is
adapted for four-class classification. The model is fine-tuned on the
AG News training data and later provides the teacher predictions
(logits) used to train the smaller BiLSTM student model.

"""

from transformers import AutoModelForSequenceClassification

class BERTTeacher :
    def __init__(self, 
                 model_name="google-bert/bert-base-uncased",
                 num_labels = 4):
        self.model_name = model_name
        self.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )

if __name__ == "__main__":
    teacher = BERTTeacher()

    print(teacher.model)
    total_params = sum(p.numel() for p in teacher.model.parameters()
)

    print(f"Total parameters: {total_params:,}")