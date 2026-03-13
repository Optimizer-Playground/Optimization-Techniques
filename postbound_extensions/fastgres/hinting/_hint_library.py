from __future__ import annotations
from ._hint import Hint
from dataclasses import dataclass, field
from bidict import bidict


@dataclass(frozen=True)
class HintLibrary:
    hint_list: list[Hint]
    hint_dict: bidict = field(init=False)
    name_dict: bidict = field(init=False)
    size: int = field(init=False)

    def __post_init__(self):
        size = len(self.hint_list)
        object.__setattr__(self, "hint_dict", bidict({i: self.hint_list[i] for i in range(size)}))
        object.__setattr__(self, "name_dict", bidict({i: self.hint_list[i].db_instr for i in range(size)}))
        object.__setattr__(self, "size", size)

    def __eq__(self, other) -> bool:
        return self.hint_list == other.hint_list
