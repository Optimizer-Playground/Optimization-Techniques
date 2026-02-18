from __future__ import annotations

from typing import overload

from ._core import CumulativeDegreeSequence, DegreeSequence
from ._piecewise_fns import PiecewiseConstantFn, PiecewiseLinearFn


@overload
def valid_compress(ds: DegreeSequence, *, accurracy: float) -> PiecewiseConstantFn: ...


@overload
def valid_compress(
    cds: CumulativeDegreeSequence, *, accurracy: float
) -> PiecewiseLinearFn: ...


def valid_compress(ds: DegreeSequence, *, accurracy: float):
    if isinstance(ds, CumulativeDegreeSequence):
        raise NotImplementedError("Compression for CDFs is not yet implemented")
    selfjoin_bound = ds.join_bound(ds)

    # TODO
    raise NotImplementedError()
