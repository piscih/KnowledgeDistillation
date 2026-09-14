import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AGNewsDataset(Dataset):

    def __init__(self, dataset, vocabulary):

        self.dataset = dataset
        self.vocabulary = vocabulary

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        text = item["text"]
        label = item["label"]

        input_ids = self.vocabulary.encode(text)

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }

def collate_fn(batch):

    input_ids = [
        item["input_ids"]
        for item in batch
    ]

    lengths = torch.tensor(
        [
            len(item["input_ids"])
            for item in batch
        ],
        dtype=torch.long
    )

    labels = torch.tensor(
        [
            item["label"]
            for item in batch
        ],
        dtype=torch.long
    )

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": labels
    }