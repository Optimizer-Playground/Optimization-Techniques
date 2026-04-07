from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import postbound as pb

from ..util import distinct_values_query, min_max_values_query, wrap_logger


def _build_text_encoder(
    column: pb.BoundColumnReference,
    dtype: str,
    *,
    database: pb.Database,
    log: pb.util.Logger,
) -> TextEncoder:
    log("Encoding", column, f"(dtype={dtype})")
    enc = TextEncoder(column, dtype)

    log("Fetching distinct values for", column)
    result_set = database.execute_query(distinct_values_query(column), raw=True)
    if not result_set:
        raise RuntimeError(f"No distinct values found for {column}")

    log("Fitting encoder for", column)
    enc.fit([row[0] for row in result_set], prepare_values=True)

    return enc


def _build_numeric_encoder(
    column: pb.BoundColumnReference,
    dtype: str,
    *,
    database: pb.Database,
    log: pb.util.Logger,
) -> NumericEncoder:
    log("Encoding", column, f"(dtype={dtype})")
    enc = NumericEncoder(column, dtype)

    log("Fetching min/max values for", column)
    min_val, max_val = database.execute_query(min_max_values_query(column), raw=True)[0]

    log("Fitting encoder for", column)
    # we still prepare_values because we might encounter a column that only contains NULL values
    enc.fit([min_val, max_val], prepare_values=True)

    return enc


def _build_datetime_encoder(
    column: pb.BoundColumnReference,
    dtype: str,
    *,
    date_only: bool,
    database: pb.Database,
    log: pb.util.Logger,
) -> DateTimeEncoder:
    log("Encoding", column, f"(dtype={dtype})")
    enc = DateTimeEncoder(column, dtype, date_only=date_only)

    log("Fetching min/max values for", column)
    min_val, max_val = database.execute_query(min_max_values_query(column), raw=True)[0]

    log("Fitting encoder for", column)
    # we still prepare_values because we might encounter a column that only contains NULL values
    enc.fit([min_val, max_val], prepare_values=True)

    return enc


class ColumnEncoder[T](ABC):
    @staticmethod
    def online(
        column: pb.BoundColumnReference,
        *,
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> ColumnEncoder:
        log = wrap_logger(verbose)

        datatype = database.schema().datatype(column)
        match datatype.lower():
            case "integer" | "smallint" | "bigint":
                enc = _build_numeric_encoder(
                    column, datatype, database=database, log=log
                )
            case "character varying" | "varchar" | "text":
                enc = _build_text_encoder(column, datatype, database=database, log=log)
            case "timestamp without time zone":
                enc = _build_datetime_encoder(
                    column,
                    datatype,
                    date_only=False,
                    database=database,
                    log=log,
                )
            case "date":
                enc = _build_datetime_encoder(
                    column,
                    datatype,
                    date_only=True,
                    database=database,
                    log=log,
                )
            case _:
                raise TypeError(
                    f"Missing encoder for dtype '{datatype}' on column {column.table.full_name}.{column.name}"
                )

        return enc

    def __new__(cls, column: pb.BoundColumnReference, dtype: str) -> ColumnEncoder:
        if cls is not ColumnEncoder:
            return super().__new__(cls)

        match dtype.lower():
            case "integer" | "smallint" | "bigint":
                return NumericEncoder(column, dtype)
            case "character varying" | "varchar" | "text":
                return TextEncoder(column, dtype)
            case "timestamp without time zone":
                return DateTimeEncoder(column, dtype, date_only=False)
            case "date":
                return DateTimeEncoder(column, dtype, date_only=True)
            case _:
                raise TypeError(
                    f"Missing encoder for dtype '{dtype}' on column {column.table.full_name}.{column.name}"
                )

    def __init__(self, column: pb.BoundColumnReference, dtype: str) -> None:
        self.column = column
        self.dtype = dtype.lower()
        self.values: np.ndarray | None = None

    @abstractmethod
    def fit(self, values: Sequence[T], *, prepare_values: bool = True) -> None: ...

    @abstractmethod
    def encode_single(self, value: T) -> np.ndarray: ...

    @abstractmethod
    def encode_batch(self, values: Iterable[T]) -> np.ndarray: ...

    @abstractmethod
    def decode(self, vec: np.ndarray) -> Optional[T]: ...

    def store(self, target_dir: Path) -> Path:
        if self.values is None:
            raise RuntimeError("Encoder not fitted yet")
        series = pd.Series(self.values.reshape(-1))
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{self.column.table.full_name}.hdf5"
        series.to_hdf(target_file, key=self.column.name, mode="a")
        return target_file

    def load(self, source_file: Path) -> None:
        series = pd.read_hdf(source_file, key=self.column.name)
        self.fit(series, prepare_values=False)

    def __repr__(self) -> str:
        return f"ColumnEncoder(column={self.column}, dtype={self.dtype})"

    def __str__(self) -> str:
        prefix = self.__repr__()
        if self.values is None:
            return f"{prefix} [not fitted]"

        n_vals = len(self.values) if self.values is not None else 0
        return f"{prefix} [fitted on {n_vals} values]"


class NumericEncoder(ColumnEncoder):
    def __init__(self, column: pb.BoundColumnReference, dtype: str) -> None:
        super().__init__(column, dtype)
        self._min_val = None
        self._max_val = None

    def fit(
        self, values: Sequence[int | float], *, prepare_values: bool = True
    ) -> None:
        arr = np.array(values).reshape(-1, 1)
        if prepare_values and arr.dtype == "object":
            # if the array is not of object type, it can't contain None values, so we can skip this step
            arr[arr == None] = np.nan  # noqa: E711
            arr = arr.astype(float)
        self.values = arr
        self._min_val = np.nanmin(arr)
        self._max_val = np.nanmax(arr)

    def encode_single(self, value: int | float) -> np.ndarray:
        if self._min_val is None or self._max_val is None:
            raise ValueError("Encoder not fitted yet")
        value = np.nan if value is None else value
        encoded = (value - self._min_val) / (self._max_val - self._min_val)
        return np.array([[encoded]])

    def encode_batch(self, values: Iterable[int | float]) -> np.ndarray:
        if self._min_val is None or self._max_val is None:
            raise ValueError("Encoder not fitted yet")
        arr = np.array([np.nan if v is None else v for v in values]).reshape(-1, 1)
        encoded = (arr - self._min_val) / (self._max_val - self._min_val)
        return encoded

    def decode(self, vec: np.ndarray) -> Optional[int | float]:
        if self._min_val is None or self._max_val is None:
            raise ValueError("Encoder not fitted yet")
        decoded = vec * (self._max_val - self._min_val) + self._min_val
        unwrapped = decoded[0][0]
        return None if np.isnan(unwrapped) else unwrapped


class TextEncoder(ColumnEncoder):
    def __init__(self, column: pb.BoundColumnReference, dtype: str) -> None:
        super().__init__(column, dtype)
        self._max_values = 0

    def fit(self, values: Sequence[str], *, prepare_values: bool = True) -> None:
        arr = np.array(values, dtype="object")
        if prepare_values:
            arr[arr == None] = "__POSTBOUND_NULL_PLACEHOLDER__"  # noqa: E711
        self.values = arr
        self._max_values = len(values)

    def encode_single(self, value: str) -> np.ndarray:
        if self.values is None:
            raise ValueError("Encoder not fitted yet")
        value = "__POSTBOUND_NULL_PLACEHOLDER__" if value is None else value
        idx = np.searchsorted(self.values, value)
        return np.array([[idx / self._max_values]])

    def encode_batch(self, values: Iterable[str]) -> np.ndarray:
        if self.values is None:
            raise ValueError("Encoder not fitted yet")
        arr = np.array(values, dtype="object")
        arr[arr == None] = "__POSTBOUND_NULL_PLACEHOLDER__"  # noqa: E711
        idxs = np.searchsorted(self.values, arr)
        return (idxs / self._max_values).reshape(-1, 1)

    def decode(self, vec: np.ndarray) -> Optional[str]:
        if self.values is None:
            raise ValueError("Encoder not fitted yet")
        idx = int(vec[0][0] * self._max_values)
        if idx < 0 or idx >= len(self.values):
            raise ValueError(f"Encoded value {vec} is out of bounds for decoder")
        unwrapped = self.values[idx][0]
        return None if unwrapped == "__POSTBOUND_NULL_PLACEHOLDER__" else unwrapped


def _timestamp(value: datetime | date | None) -> float:
    match value:
        case datetime():
            return value.timestamp()
        case date():
            return datetime.combine(value, time()).timestamp()
        case None:
            return np.nan


class DateTimeEncoder(ColumnEncoder):
    def __new__(
        cls, column: pb.BoundColumnReference, dtype: str, *, date_only: bool
    ) -> DateTimeEncoder:
        return super(object).__new__(cls)

    def __init__(
        self, column: pb.BoundColumnReference, dtype: str, *, date_only: bool
    ) -> None:
        super().__init__(column, dtype)
        self.date_only = date_only
        self._min_val = None
        self._max_val = None

    def fit(
        self, values: Sequence[datetime | date], *, prepare_values: bool = True
    ) -> None:
        arr = np.array([_timestamp(v) for v in values]).reshape(-1, 1)
        self.values = arr
        self._min_val = np.nanmin(arr)
        self._max_val = np.nanmax(arr)

    def encode_single(self, value: datetime | date) -> np.ndarray:
        if self._min_val is None or self._max_val is None:
            raise ValueError("Encoder not fitted yet")
        timestamp = _timestamp(value)
        encoded = (timestamp - self._min_val) / (self._max_val - self._min_val)
        return np.array([[encoded]])

    def encode_batch(self, values: Iterable[datetime | date]) -> np.ndarray:
        if self._min_val is None or self._max_val is None:
            raise ValueError("Encoder not fitted yet")
        arr = np.array([_timestamp(v) for v in values]).reshape(-1, 1)
        encoded = (arr - self._min_val) / (self._max_val - self._min_val)
        return encoded

    def decode(self, vec: np.ndarray) -> Optional[datetime | date]:
        if self._min_val is None or self._max_val is None:
            raise ValueError("Encoder not fitted yet")
        decoded = vec * (self._max_val - self._min_val) + self._min_val
        timestamp = decoded[0][0]
        if np.isnan(timestamp):
            return None
        dt = datetime.fromtimestamp(timestamp)
        return dt.date() if self.date_only else dt
