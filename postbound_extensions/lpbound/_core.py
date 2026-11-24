from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

NormSet = Sequence[float]


@dataclass
class LpBoundSpec:
    norms: NormSet = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf])
    num_mcvs: int = 5000
    hist_buckets: int = 128

    @staticmethod
    def default() -> LpBoundSpec:
        """The default LpBound configuration, as used in the paper.

        The values are as follows:

        - `norms` = {1, ..., 10} plus the infinity norm
        - `num_mcvs` = 5000
        - `hist_buckets` = 128
        """
        return LpBoundSpec()
