from __future__ import annotations

from collections.abc import Generator, Sequence

from ._piecewise_fns import DegreeSequence, PiecewiseLinearFn, Segment


def valid_compress(ds: DegreeSequence, *, accuracy: float) -> PiecewiseLinearFn:
    """Compresses a raw degree sequence into a piecewise linear function.

    This is basically a 1:1 mapping of the *ValidCompress* algorithm from the original SafeBound
    paper. The `accuracy` controls how much error/overestimation is allowed in the resulting
    segements. It corresponds to the *c* parameter from the paper.
    """
    if not ds:
        return PiecewiseLinearFn(slopes=[], intercepts=[], bounds=[], column=ds.column)

    selfjoin_bound = ds.join_bound(ds)
    err_threshold = accuracy * selfjoin_bound

    error = 0
    segments: list[Segment] = [Segment.initial(ds.max_deg)]
    for i in range(ds.distinct_values):
        deg = ds[i]
        slope = segments[-1].slope
        error += (slope**2) * (deg / slope) - (deg**2)

        if error >= err_threshold:
            next_segment = Segment.after(segments[-1], slope=deg)
            segments.append(next_segment)
            error = 0

        segments[-1].higher += deg / segments[-1].slope

    return PiecewiseLinearFn.from_segments(segments, column=ds.column)


def generate_compressed_intervals[T](
    frequencies: Sequence[tuple[T, int]], accuracy: float
) -> Generator[Sequence[T], None, None]:
    """Runs `valid_compress` over a degree sequence, but yields the intervals of values that are compressed together.

    The degree sequence is constructed from the `frequencies`.

    This is basically a 1:1 mapping of `valid_compress`. However, instead of actually compressing the segments,
    we provide the values that are compressed together, starting with the highest frequency values.
    """
    if not frequencies:
        return

    frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)
    selfjoin_bound = sum(deg**2 for _, deg in frequencies)
    err_threshold = accuracy * selfjoin_bound

    error = 0
    current_interval = []
    slope = frequencies[0][1]
    for val, deg in frequencies:
        error += (slope**2) * (deg / slope) - (deg**2)

        if error >= err_threshold:
            yield current_interval

            current_interval = []
            slope = deg
            error = 0

        current_interval.append(val)

    yield current_interval
