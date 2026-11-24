from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

NormSet = Sequence[float]
"""The Lp norms that are computed for each join key to calculate the join upper bound."""


@dataclass
class LpBoundSpec:
    """ "Hyperparameters" for the LpBound catalog.

    - `norms` are the Lp norms that should be computed per join key.
    - `num_mcvs` are the number of most-common-values per equality filter column to condition Lp norms.
    - `hist_buckets` are the maximum number of buckets each range filter histogram should have. The actual histograms
      form a hierarchy where each histogram has 1/2 the number of buckets of its parent. This results in a total of
      2n - 1 conditioned Lp norms per range filter column.
    """

    norms: NormSet = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf])
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
