from __future__ import annotations

from ._piecewise_fns import DegreeSequence, PiecewiseLinearFn, Segment


def valid_compress(ds: DegreeSequence, *, accuracy: float) -> PiecewiseLinearFn:
    """Compresses a raw degree sequence into a piecewise linear function.

    This is basically a 1:1 mapping of the *ValidCompress* algorithm from the original SafeBound
    paper. The `accuracy` controls how much error/overestimation is allowed in the resulting
    segements. It corresponds to the *c* parameter from the paper.
    """
    selfjoin_bound = ds.join_bound(ds)
    err_threshold = accuracy * selfjoin_bound

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

    return PiecewiseLinearFn.from_segments(segments, column=ds.column)
