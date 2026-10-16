import torch.nn as nn
from common import PAD_ID


class RecurrentLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        cell="lstm",
        emb_dim=256,
        hid_dim=512,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        self.cfg = dict(
            vocab_size=vocab_size,
            cell=cell,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        rnn_drop = dropout if num_layers > 1 else 0.0
        if cell == "rnn":
            self.rnn = nn.RNN(
                emb_dim,
                hid_dim,
                num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=rnn_drop,
            )
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                emb_dim, hid_dim, num_layers, batch_first=True, dropout=rnn_drop
            )
        else:
            raise ValueError(cell)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, hidden=None):
        o, h = self.rnn(self.drop(self.emb(x)), hidden)
        return self.out(self.drop(o)), h

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
