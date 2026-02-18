from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import postbound as pb
from numpy.typing import NDArray


class DegreeSequence:
    @staticmethod
    def from_mcv(mcv: pb.db.MostCommonValues) -> DegreeSequence:
        return DegreeSequence(mcv.frequencies)

    def __init__(
        self, degrees: Iterable[int | pb.Cardinality] | NDArray[np.int_]
    ) -> None:
        if not isinstance(degrees, np.ndarray):
            degrees = [
                int(deg)
                for deg in degrees
                if not isinstance(deg, pb.Cardinality) or deg.is_valid()
            ]
            degrees = np.asarray(degrees)
        degrees = np.sort(degrees)[::-1]  # see https://stackoverflow.com/q/26984414
        self._degrees: NDArray[np.int_] = np.array(degrees)

    @property
    def max_deg(self) -> int:
        return self._degrees[0]

    @property
    def min_deg(self) -> int:
        return self._degrees[-1]

    @property
    def max_freq(self) -> int:
        return self._degrees[0]

    @property
    def degrees(self) -> Sequence[int]:
        return self._degrees.tolist()

    @property
    def distinct_values(self) -> int:
        return len(self)

    @property
    def cardinality(self) -> int:
        return int(np.sum(self._degrees))

    def join_bound(self, other: DegreeSequence) -> int:
        return int(np.sum(self._degrees * other._degrees))

    def as_cdf(self) -> CumulativeDegreeSequence:
        return CumulativeDegreeSequence(self._degrees)

    def __len__(self) -> int:
        return len(self._degrees)

    def __getitem__(self, i) -> int:
        return self._degrees[i]

    def __le__(self, other: DegreeSequence) -> bool:
        if len(self) != len(other):
            raise ValueError(
                "Degree sequences must have the same length for comparison"
            )
        return bool(np.min(other._degrees - self._degrees) >= 0)

    def __repr__(self) -> str:
        degrees = repr(self._degrees.tolist())
        return f"DegreeSequence({degrees})"

    def __str__(self) -> str:
        return str(self._degrees.tolist())


class CumulativeDegreeSequence:
    @staticmethod
    def from_mcv(mcv: pb.db.MostCommonValues) -> CumulativeDegreeSequence:
        return CumulativeDegreeSequence(mcv.frequencies)

    def __init__(
        self, degrees: Iterable[int | pb.Cardinality] | NDArray[np.int_]
    ) -> None:
        if not isinstance(degrees, np.ndarray):
            degrees = [
                int(deg)
                for deg in degrees
                if not isinstance(deg, pb.Cardinality) or deg.is_valid()
            ]
            degrees = np.asarray(degrees)
        degrees[::-1].sort()  # type: ignore - see https://stackoverflow.com/q/26984414
        self._degrees = np.cumsum(degrees)

    @property
    def max_deg(self) -> int:
        return self._degrees[0]

    @property
    def degrees(self) -> Sequence[int]:
        return self._degrees.tolist()

    @property
    def distinct_values(self) -> int:
        return len(self)

    @property
    def cardinality(self) -> int:
        return np.sum(self._degrees)

    def as_ds(self) -> DegreeSequence:
        pseudo_derivative = np.diff(self._degrees)
        degs = np.concat([self._degrees[0:1], pseudo_derivative])
        return DegreeSequence(degs)

    def __len__(self) -> int:
        return len(self._degrees)

    def __getitem__(self, i) -> int:
        return self._degrees[i]

    def __le__(self, other: CumulativeDegreeSequence) -> bool:
        if len(self) != len(other):
            raise ValueError(
                "Degree sequences must have the same length for comparison"
            )
        return np.min(other._degrees - self._degrees) >= 0

    def __repr__(self) -> str:
        degrees = repr(self._degrees.tolist())
        return f"CumulativeDegreeSequence({degrees})"

    def __str__(self) -> str:
        return str(self._degrees.tolist())
