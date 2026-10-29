import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional

# ============================================================================
# BASELINE NMT MODEL (WITHOUT ATTENTION)
# ============================================================================

class BaselineEncoder(nn.Module):
    """
    Standard Encoder without Attention:
    English Sentence -> Embedding -> Encoder LSTM -> Final Hidden & Cell States
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embed_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, src_len, hidden_dim]
        # hidden, cell: [num_layers, batch_size, hidden_dim]
        return outputs, (hidden, cell)


class BaselineDecoder(nn.Module):
    """
    Standard Decoder without Attention:
    Takes previous hidden state & target token -> LSTM -> Dense + Softmax -> Hindi token
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # input_token: [batch_size]
        input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # output: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, vocab_size]
        return prediction, (hidden, cell)


class Seq2SeqBaseline(nn.Module):
    """
    End-to-end Baseline Encoder-Decoder Model without Attention
    """
    def __init__(self, encoder: BaselineEncoder, decoder: BaselineDecoder, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)
        _, (hidden, cell) = self.encoder(src)

        # First token fed to decoder is <SOS> (tgt[:, 0])
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            prediction, (hidden, cell) = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = prediction

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    def translate_sentence(self, src: torch.Tensor, max_len: int = 50, sos_idx: int = 1, eos_idx: int = 2):
        """
        Greedy decoding at inference time (no teacher forcing)
        """
        self.eval()
        with torch.no_grad():
            _, (hidden, cell) = self.encoder(src)
            input_token = torch.tensor([sos_idx], device=src.device)
            decoded_indices = []

            for _ in range(max_len):
                prediction, (hidden, cell) = self.decoder(input_token, hidden, cell)
                top1 = prediction.argmax(1).item()
                if top1 == eos_idx:
                    break
                decoded_indices.append(top1)
                input_token = torch.tensor([top1], device=src.device)

        return decoded_indices


# ============================================================================
# ATTENTION-BASED NMT MODEL (BAHDANAU ADDITIVE ATTENTION)
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Additive Attention Mechanism (Bahdanau et al., 2014):
    e_{t,i} = v_a^T tanh(W_s * s_{t-1} + W_h * h_i)
    alpha_{t,i} = exp(e_{t,i}) / sum_j exp(e_{t,j})
    c_t = sum_i alpha_{t,i} * h_i
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: [batch_size, hidden_dim] (s_{t-1})
        # encoder_outputs: [batch_size, src_len, hidden_dim] (h_1, ..., h_n)
        # src_mask: [batch_size, src_len], 1 for non-pad, 0 for pad
        src_len = encoder_outputs.shape[1]

        # Expand decoder_hidden across src_len: [batch_size, src_len, hidden_dim]
        s_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # e_{t,i} = v_a^T * tanh(W_s * s_{t-1} + W_h * h_i)
        energy = torch.tanh(self.W_s(s_expanded) + self.W_h(encoder_outputs))  # [batch_size, src_len, hidden_dim]
        scores = self.v_a(energy).squeeze(2)  # [batch_size, src_len]

        if src_mask is not None:
            # Mask out padding tokens with a large negative value
            scores = scores.masked_fill(~src_mask, -1e9)

        # Attention weights: alpha_{t,i}
        alpha = F.softmax(scores, dim=1)  # [batch_size, src_len]

        # Context vector: c_t = sum_i alpha_{t,i} * h_i
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, hidden_dim]

        return context, alpha


class AttentionEncoder(nn.Module):
    """
    Encoder for Attention Model:
    Outputs all hidden states h_1, h_2, ..., h_n to be retained for attention computation.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        encoder_outputs, (hidden, cell) = self.lstm(embedded)
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        return encoder_outputs, (hidden, cell)


class AttentionDecoder(nn.Module):
    """
    Decoder with Attention:
    Takes previous hidden state s_{t-1}, computes context vector c_t,
    concatenates [embedded_token; c_t] into LSTM, and projects to vocab logits.
    Inspectable attention weights alpha_t are extracted at each step.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = BahdanauAttention(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # LSTM takes concatenation of [embedded token (embed_dim); context vector (hidden_dim)]
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection combines [s_t; c_t; embedded_token]
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + embed_dim, vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # input_token: [batch_size]
        # hidden: [num_layers, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]

        # Use top layer decoder hidden state for computing attention
        s_prev = hidden[-1]  # [batch_size, hidden_dim]
        context, alpha = self.attention(s_prev, encoder_outputs, src_mask)

        input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, embed_dim]

        # Concatenate embedded token and context vector: [batch_size, 1, embed_dim + hidden_dim]
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        s_t = output.squeeze(1)  # [batch_size, hidden_dim]
        emb_squeezed = embedded.squeeze(1)  # [batch_size, embed_dim]

        # Linear projection to vocabulary logits
        prediction = self.fc_out(torch.cat((s_t, context, emb_squeezed), dim=1))  # [batch_size, vocab_size]

        return prediction, (hidden, cell), alpha


class Seq2SeqAttention(nn.Module):
    """
    Complete Attention-Based Seq2Seq Architecture
    English -> Embedding -> Encoder LSTM -> [h1..hn] -> Attention -> Context -> Decoder LSTM -> Dense+Softmax -> Hindi
    """
    def __init__(self, encoder: AttentionEncoder, decoder: AttentionDecoder, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx)  # True for valid tokens, False for <PAD>

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)
        attentions = torch.zeros(batch_size, tgt_len, src_len, device=src.device)

        src_mask = self.create_mask(src)
        encoder_outputs, (hidden, cell) = self.encoder(src)

        input_token = tgt[:, 0]  # <SOS>

        for t in range(1, tgt_len):
            prediction, (hidden, cell), alpha = self.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            outputs[:, t, :] = prediction
            attentions[:, t, :] = alpha

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs, attentions

    def translate_sentence(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> Tuple[list, torch.Tensor]:
        """
        Inference with greedy decoding and full attention weights matrix extraction.
        Returns:
            decoded_indices: List of generated token IDs
            attention_matrix: [tgt_generated_len, src_len]
        """
        self.eval()
        with torch.no_grad():
            src_mask = self.create_mask(src)
            encoder_outputs, (hidden, cell) = self.encoder(src)

            input_token = torch.tensor([sos_idx], device=src.device)
            decoded_indices = []
            attention_weights_list = []

            for _ in range(max_len):
                prediction, (hidden, cell), alpha = self.decoder(
                    input_token, hidden, cell, encoder_outputs, src_mask
                )
                attention_weights_list.append(alpha.squeeze(0).cpu())
                top1 = prediction.argmax(1).item()

                if top1 == eos_idx:
                    break

                decoded_indices.append(top1)
                input_token = torch.tensor([top1], device=src.device)

            if attention_weights_list:
                attention_matrix = torch.stack(attention_weights_list, dim=0)  # [len(decoded), src_len]
            else:
                attention_matrix = torch.zeros(0, src.shape[1])

        return decoded_indices, attention_matrix
