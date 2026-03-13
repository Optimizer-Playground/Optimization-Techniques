from __future__ import annotations
import numpy as np
from ._hint_library import HintLibrary
from ._hint import Hint
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HintSet:
    hs_int: int
    library: HintLibrary
    hs_bin: list[int] = field(init=False)
    _iterlist: list[tuple[str, str, bool]] = field(init=False)

    def __post_init__(self):
        if isinstance(self.hs_int, np.integer):
            object.__setattr__(self, "hs_int", int(self.hs_int))

        if not isinstance(self.hs_int, int):
            raise ValueError(f'Input {self.hs_int} is of type {type(self.hs_int)} not int')

        if not 0 <= self.hs_int < 2 ** self.library.size:
            raise ValueError(f"Hint Set Integer: {self.hs_int} out of bounds for {self.library.size} hints")

        hs_bin = self._binary()
        object.__setattr__(self, "hs_bin", hs_bin)
        for i, hint in self.library.hint_dict.items():
            object.__setattr__(self, hint.name, bool(hs_bin[i]))

        object.__setattr__(
            self,
            "_iterlist",
            [(self.get_hint(i).db_instr, self.bool_rep[i]) for i in range(self.library.size)]
        )

    def __repr__(self):
        return f"HS_{self.hs_int}"

    @classmethod
    def from_hints(cls, set_of_hints: set[Hint], library: HintLibrary) -> HintSet:
        # hints set takes names of the form presented in postgres_hints.py
        int_rep = sum([2 ** library.hint_dict.inverse[hint_name] for hint_name in set_of_hints])
        return cls(int_rep, library)

    @classmethod
    def from_disabled_set(cls, instructions_to_disable: set[str], library: HintLibrary) -> HintSet:
        # takes a set of names and disables them from the default setting (all on) the names have the pg format
        int_rep = (2**library.size - 1) - sum([2 ** library.name_dict.inverse[hint_name] for hint_name in instructions_to_disable])
        return cls(int_rep, library)

    @classmethod
    def from_int_set(cls, int_set: set[int], library: HintLibrary) -> HintSet:
        # int set takes integers in the form of 2**i
        return cls(sum(int_set), library)

    def __iter__(self):
        return iter(self._iterlist)

    @property
    def n_disabled(self):
        return self.hs_bin.count(0)

    @property
    def n_enabled(self):
        return self.hs_bin.count(1)

    @property
    def bool_rep(self) -> list[bool]:
        return [bool(i) for i in self.hs_bin]

    @property
    def bin_rep(self) -> list[int]:
        return self.hs_bin

    @property
    def int_rep(self):
        return self.hs_int

    def _binary(self):
        # lowest left to right
        return np.array(list(np.binary_repr(self.hs_int, width=self.library.size)), dtype=int)[::-1].tolist()

    def get_hint_name(self, index: int) -> str:
        return self.get_hint(index).name

    def get_hint(self, index: int) -> Hint:
        if not -1 < index < self.library.size:
            raise ValueError(f"Index {index} is out of bounds for {self.library.size} hints")
        return self.library.hint_dict[index]
