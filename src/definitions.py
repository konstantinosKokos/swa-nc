from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass(unsafe_hash=True)
class Nomino:
    entry: str
    definition: str
    subject_concord: str

    @staticmethod
    def from_json(fp: str | Path) -> list[Nomino]:
        with open(fp, 'r') as f:
            return [Nomino(**entry) for entry in json.load(f)]

    def to_json(self) -> dict[str, str]:
        return asdict(self)
