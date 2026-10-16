import json, random, re
from pathlib import Path
import numpy as np
import torch

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

ROOT = Path(__file__).resolve().parent
DATA, CKPT, RES, HUM = (
    ROOT / "processed",
    ROOT / "checkpoints",
    ROOT / "results",
    ROOT / "human_eval",
)
for _d in (DATA, CKPT, RES, HUM):
    _d.mkdir(exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    try:
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:\"()\-]")


def clean(text: str) -> str:
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str):
    return _TOKEN_RE.findall(clean(text))


def detokenize(tokens):
    text = " ".join(t for t in tokens if t not in (PAD, BOS, EOS))
    text = re.sub(r"\s+([.,!?;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(
        r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
    )
    text = re.sub(r"\bi\b", "I", text)
    return text


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos))

    @staticmethod
    def load(path):
        return Vocab(json.loads(Path(path).read_text()))
