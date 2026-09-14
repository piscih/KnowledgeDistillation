import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, input_ids, lengths):

        embeddings = self.embedding(input_ids)

        packed_embeddings = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        outputs, (hidden, cell) = self.lstm(
            packed_embeddings
        )

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        hidden_state = torch.cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )

        hidden_state = self.dropout(
            hidden_state
        )

        logits = self.classifier(
            hidden_state
        )

        return logits