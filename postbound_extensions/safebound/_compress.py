from __future__ import annotations

from ._core import DegreeSequence
from ._piecewise_fns import PiecewiseLinearFn, Segment


def valid_compress(ds: DegreeSequence, *, accurracy: float) -> PiecewiseLinearFn:
    selfjoin_bound = ds.join_bound(ds)
    err_threshold = accurracy * selfjoin_bound

    error = 0
    segments: list[Segment] = [Segment.initial(ds.max_deg)]
    for i in range(ds.distinct_values):
        deg = ds[i]
        slope = segments[-1].slope
        error += (
            (slope**2) * (deg / slope)  #
            - (deg**2)  #
        )

        if error >= err_threshold:
            next_segment = Segment.after(segments[-1], slope=deg)
            segments.append(next_segment)
            error = 0

        segments[-1].higher += deg / segments[-1].slope

    return PiecewiseLinearFn(segments, num_distinct=ds.distinct_values)
