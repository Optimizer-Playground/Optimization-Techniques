from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Hint:
    name: str
    db_instr: str
    db_instr_val: bool
