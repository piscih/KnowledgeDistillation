import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AGNewsKDDataset(Dataset):
    """
    Dataset for Knowledge Distillation.

    For each AG News example, this dataset prepares:
        1. Input for the BiLSTM student
        2. Input for the BERT teacher
        3. Attention mask for BERT
        4. Ground-truth label
        5. Original dataset index
    """

    def __init__(
        self,
        dataset,
        vocabulary,
        tokenizer,
        max_length=128,
    ):
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        text = item["text"]
        label = item["label"]

        # ----------------------------------------------------
        # BiLSTM input
        # ----------------------------------------------------

        student_input_ids = self.vocabulary.encode(text)

        # ----------------------------------------------------
        # BERT input
        # ----------------------------------------------------

        teacher_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        teacher_input_ids = teacher_encoding["input_ids"]

        teacher_attention_mask = teacher_encoding[
            "attention_mask"
        ]

        # ----------------------------------------------------
        # Return everything
        # ----------------------------------------------------

        return {
            "index": index,

            "student_input_ids": torch.tensor(
                student_input_ids,
                dtype=torch.long,
            ),

            "teacher_input_ids": torch.tensor(
                teacher_input_ids,
                dtype=torch.long,
            ),

            "teacher_attention_mask": torch.tensor(
                teacher_attention_mask,
                dtype=torch.long,
            ),

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


def collate_fn(batch):

    # ========================================================
    # Student inputs
    # ========================================================

    student_input_ids = [
        item["student_input_ids"]
        for item in batch
    ]

    student_lengths = torch.tensor(
        [
            len(item["student_input_ids"])
            for item in batch
        ],
        dtype=torch.long,
    )

    student_input_ids = pad_sequence(
        student_input_ids,
        batch_first=True,
        padding_value=0,
    )


    # ========================================================
    # Teacher inputs
    # ========================================================

    teacher_input_ids = [
        item["teacher_input_ids"]
        for item in batch
    ]

    teacher_attention_masks = [
        item["teacher_attention_mask"]
        for item in batch
    ]

    teacher_input_ids = pad_sequence(
        teacher_input_ids,
        batch_first=True,
        padding_value=0,
    )

    teacher_attention_mask = pad_sequence(
        teacher_attention_masks,
        batch_first=True,
        padding_value=0,
    )


    # ========================================================
    # Labels
    # ========================================================

    labels = torch.tensor(
        [
            item["label"]
            for item in batch
        ],
        dtype=torch.long,
    )


    # ========================================================
    # Dataset indices
    # ========================================================

    indices = torch.tensor(
        [
            item["index"]
            for item in batch
        ],
        dtype=torch.long,
    )


    # ========================================================
    # Return batch
    # ========================================================

    return {
        "indices": indices,

        "student_input_ids": student_input_ids,

        "student_lengths": student_lengths,

        "teacher_input_ids": teacher_input_ids,

        "teacher_attention_mask": teacher_attention_mask,

        "labels": labels,
    }