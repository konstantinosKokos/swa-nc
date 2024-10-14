from definitions import Nomino
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
import pickle
from collections import Counter


def make_cfn(device: str, padding_value: int = 0):
    def cfn(xs: list[tuple[list[int], int]]) -> tuple[Tensor, Tensor]:
        ys = torch.tensor([y for _, y in xs]).to(device)
        xs = pad_sequence([torch.tensor(x) for x, _ in xs], batch_first=True, padding_value=padding_value).to(device)
        return xs, ys
    return cfn


def encode(tokenizer: BertTokenizer, string: str) -> list[int]:
    return tokenizer.encode(string)


def get_tokenizer(id: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(id)


if __name__ == '__main__':
    tokenizer = get_tokenizer('sentence-transformers/all-MiniLM-L12-v2')

    nominos = Nomino.from_json('../data/parsed.json')
    c = Counter(nom.subject_concord for nom in nominos)
    print(c)
    classes = sorted([k for k, v in c.items() if v > 80], key=lambda k: c[k], reverse=True)

    print(f'RB: {sum( (c[k0] / sum(c[k] for k in classes)) ** 2 for k0 in classes )}')

    nominos = [n for n in nominos if n.subject_concord in classes]
    tokenized = [(nom,
                  encode(tokenizer, nom.definition),
                  classes.index(nom.subject_concord)) for nom in nominos]

    with open('../data/tokenized.p', 'wb') as f:
        pickle.dump((tokenized, classes), f)
