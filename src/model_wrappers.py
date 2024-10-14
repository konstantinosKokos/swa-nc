from torch import Tensor
from torch.nn import Module, Linear
from torch.nn.functional import cross_entropy, dropout, normalize
from torch.optim import Optimizer

from transformers import BertModel

from typing import Iterable


class SBERT(Module):
    def __init__(self, core: str = 'sentence-transformers/all-MiniLM-L12-v2'):
        super().__init__()
        self.bert = BertModel.from_pretrained(core)
        self.eval()

    def forward(self, xs: Tensor) -> Tensor:
        atn_mask = xs.ne(0)
        xs = self.bert(xs, attention_mask=atn_mask)[0]
        xs = dropout(xs, p=0.2, training=self.training) * atn_mask[:, :, None]
        xs = xs.sum(dim=-2) / atn_mask.sum(-1, keepdim=True)
        return normalize(xs, p=2, dim=-1)


class Classifier(Module):
    def __init__(self, num_classes: int, freeze: bool, core: str = 'sentence-transformers/all-MiniLM-L12-v2'):
        super().__init__()
        self.core = SBERT(core)
        if freeze:
            for p in self.core.parameters():
                p.requires_grad = False
        self.linear = Linear(384, num_classes)

    def forward(self, xs: Tensor) -> Tensor:
        xs = self.core(xs)
        return self.linear(xs)

    def go_batch(self, xs: Tensor, ys: Tensor, optimizer: Optimizer | None) -> tuple[Tensor, Tensor]:
        out = self.forward(xs)
        loss = cross_entropy(out, ys, reduction='mean')
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.detach(), out

    def train_batch(self, xs: Tensor, ys: Tensor, optimizer: Optimizer) -> tuple[Tensor, Tensor]:
        return self.go_batch(xs, ys, optimizer)

    def eval_batch(self, xs: Tensor, ys: Tensor) -> tuple[Tensor, Tensor]:
        return self.go_batch(xs, ys, None)

    def go_epoch(
            self,
            data: Iterable[tuple[Tensor, Tensor]],
            optimizer: Optimizer | None
    ) -> tuple[float, list[int], list[int]]:
        predictions, truth, epoch_loss = [], [], 0.
        for xs, ys in data:
            loss, ps = self.go_batch(xs, ys, optimizer)
            epoch_loss += loss.item()
            predictions += ps.argmax(-1).cpu().tolist()
            truth += ys.cpu().tolist()
        return epoch_loss, predictions, truth

    def train_epoch(
            self,
            data: Iterable[tuple[Tensor, Tensor]],
            optimizer: Optimizer
    ) -> tuple[float, list[int], list[int]]:
        return self.go_epoch(data, optimizer)

    def eval_epoch(self, data: Iterable[tuple[Tensor, Tensor]]) -> tuple[float, list[int], list[int]]:
        return self.go_epoch(data, None)
