"""Small one-to-one mapping used by FASTgres hints and FASTlabel candidates."""

from collections.abc import Hashable, Iterator, Mapping, MutableMapping
from types import MappingProxyType


class BidirectionalMap[K: Hashable, V: Hashable](MutableMapping[K, V]):
    """A mutable mapping with unique values and a live, read-only inverse view.

    Assigning an existing key replaces its value and removes the old inverse
    entry. Assigning a value owned by another key raises ``ValueError`` without
    changing the mapping. Reassigning an identical pair is a no-op.

    This implements the subset of ``bidict`` used by FASTgres. All mutations go
    through the forward mapping; ``inverse`` and its alias ``inv`` are for lookup.
    """

    def __init__(self, entries: Mapping[K, V] | None = None) -> None:
        self._forward: dict[K, V] = {}
        self._backward: dict[V, K] = {}
        if entries is not None:
            self.update(entries)

    def __getitem__(self, key: K) -> V:
        return self._forward[key]

    def __setitem__(self, key: K, value: V) -> None:
        has_key = key in self._forward
        if value in self._backward:
            owner = self._backward[value]
            if owner is key or owner == key:
                return
            raise ValueError(f"Value {value!r} already belongs to key {owner!r}")
        if has_key:
            del self._backward[self._forward[key]]
        self._forward[key] = value
        self._backward[value] = key

    def __delitem__(self, key: K) -> None:
        value = self._forward.pop(key)
        del self._backward[value]

    def __iter__(self) -> Iterator[K]:
        return iter(self._forward)

    def __len__(self) -> int:
        return len(self._forward)

    @property
    def inverse(self) -> Mapping[V, K]:
        return MappingProxyType(self._backward)

    @property
    def inv(self) -> Mapping[V, K]:
        return self.inverse

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._forward!r})"
