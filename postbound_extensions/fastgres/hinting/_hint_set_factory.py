from ._hint_library import HintLibrary
from ._hint_set import HintSet
from dataclasses import dataclass


@dataclass
class HintSetFactory:

    hint_library: HintLibrary

    def __post_init__(self):

        if self.hint_library.size <= 0:
            raise ValueError("Hint library size must be greater than zero")

    @property
    def size(self):
        return self.hint_library.size

    @property
    def hint_names(self):
        return [hint.name for hint in self.hint_library.hint_list]

    def hint_set(self, hint_set_int: int):
        return HintSet(hint_set_int, self.hint_library)

    def hint_set_from_disabled(self, disabled_instructions: set[str]):
        return HintSet.from_disabled_set(disabled_instructions, self.hint_library)

    def default_hint_set(self):
        return self.hint_set(2**self.hint_library.size-1)
