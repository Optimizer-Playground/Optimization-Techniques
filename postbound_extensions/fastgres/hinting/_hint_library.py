from __future__ import annotations

from dataclasses import dataclass, field

from .._mapping import BidirectionalMap
from ._hint import Hint


@dataclass(frozen=True)
class HintLibrary:
    hint_list: list[Hint]
    hint_dict: BidirectionalMap[int, Hint] = field(init=False)
    name_dict: BidirectionalMap[int, str] = field(init=False)
    size: int = field(init=False)

    def __post_init__(self):
        size = len(self.hint_list)
        object.__setattr__(self, "hint_dict", BidirectionalMap({i: self.hint_list[i] for i in range(size)}))
        object.__setattr__(self, "name_dict", BidirectionalMap({i: self.hint_list[i].db_instr for i in range(size)}))
        object.__setattr__(self, "size", size)

    def __eq__(self, other) -> bool:
        return self.hint_list == other.hint_list
