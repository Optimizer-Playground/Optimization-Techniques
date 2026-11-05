from dataclasses import dataclass

from ._hint_library import HintLibrary
from ._hint_set import HintSet


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
        """Return the all-enabled starting point used by FastLabel's search."""
        return self.hint_set(2**self.hint_library.size-1)

    def database_default_hint_set(self):
        """Return the defaults declared by the library.

        For example, a PG19 library with miscellaneous hints starts with JIT off.
        Hint bits still encode actual on/off values; their meaning is unchanged.
        """
        defaults = sum(1 << i for i, hint in self.hint_library.hint_dict.items() if hint.db_instr_val)
        return self.hint_set(defaults)
