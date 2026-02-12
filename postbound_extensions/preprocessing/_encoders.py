from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import postbound as pb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from ..util import distinct_values_query, wrap_logger


class ColumnEncoder:
    @staticmethod
    def online(
        column: pb.ColumnReference,
        *,
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> ColumnEncoder:
        log = wrap_logger(verbose)
        log("Encoding", column)

        datatype = database.schema().datatype(column)
        if (table := column.table) is None:
            raise ValueError("Column must be bound to a table")
        enc = ColumnEncoder(column.name, table.full_name, datatype)

        log("Fetching distinct values for", column)
        values = database.execute_query(distinct_values_query(column))

        log("Fitting encoder for", column)
        enc.fit(values)

        return enc

    def __init__(self, colname: str, table: str, dtype: str) -> None:
        self.colname = colname
        self.table = table
        self.dtype = dtype
        self._encoder = None
        self._values: np.ndarray | None = None

    def fit(self, values: Sequence, *, prepare_values: bool = True) -> None:
        if not len(values):
            return

        encoder = self._determine_encoder(self.dtype)
        if prepare_values:
            values = self._prepare_values(values)
        values = np.array(values).reshape(-1, 1)

        encoder.fit(values)
        self._encoder = encoder
        self._values = values

    def encode_single(self, value: object) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("Encoder not fitted yet")

        value = self._null_handler(value)
        value = self._cast(value)
        return self._encoder.transform([[value]])

    def encode_batch(self, values: Iterable) -> np.ndarray:
        encs = [self.encode_single(value) for value in values]
        return np.concat(encs)

    def decode(self, vec: np.ndarray) -> object:
        if self._encoder is None:
            raise RuntimeError("Encoder not fitted yet")

        raw = self._encoder.inverse_transform([vec])
        return self._unwrap_value(raw[0])

    def store(self, target_dir: Path) -> Path:
        series = pd.Series(self._values.reshape(-1))
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{self.table}.hdf5"
        series.to_hdf(target_file, key=self.colname, mode="a")
        return target_file

    def load(self, target_file: Path) -> None:
        series = pd.read_hdf(target_file, key=self.colname)
        self.fit(series, prepare_values=False)

    def _cast(self, value: object) -> object:
        if self.dtype == "timestamp without time zone" and isinstance(value, str):
            return datetime.fromisoformat(value).timestamp()
        if self.dtype == "date" and isinstance(value, str):
            # Postgres allows dates like '2024-01-01 00:00:00' which are not ISO format for dates
            # At the same time, datetime.fromisoformat can parse dates like '2024-01-01' as datetime objects
            # Therefore, we just delegate all parsing to datetime.fromisoformat and convert to timestamp
            return datetime.fromisoformat(value).timestamp()

        return value

    def _null_handler(self, value: object) -> object:
        if value is not None:
            return value

        match self.dtype:
            case "integer" | "smallint" | "bigint":
                return np.nan
            case "character varying" | "text":
                return "__POSTBOUND_NULL_PLACEHOLDER__"
            case "timestamp without time zone":
                return np.nan
            case "date":
                return np.nan
            case _:
                raise TypeError(
                    f"Missing null handler for dtype '{self.dtype}' on column {self.table}.{self.colname}"
                )

    def _unwrap_value(self, value: object) -> Optional[object]:
        print("Checking", value)
        match self.dtype:
            case "integer" | "smallint" | "bigint":
                return None if np.isnan(value) else value
            case "character varying" | "text":
                return None if value == "__POSTBOUND_NULL_PLACEHOLDER__" else value
            case "timestamp without time zone" if np.isnan(value):
                return None
            case "timestamp without time zone" if not np.isnan(value):
                return datetime.fromtimestamp(value)
            case "date" if np.isnan(value):
                return None
            case "date" if not np.isnan(value):
                return date.fromtimestamp(value)
            case _:
                raise TypeError(
                    f"Missing unwrapping logic for dtype '{self.dtype}' on column {self.table}.{self.colname}"
                )

    def _replace_nulls(self, values: Sequence) -> Sequence:
        return [self._null_handler(v) if v is None else v for v in values]

    def _determine_encoder(self, dtype: str):
        match self.dtype:
            case "integer" | "smallint" | "bigint":
                return MinMaxScaler()
            case "character varying" | "text":
                return make_pipeline(
                    OrdinalEncoder(),
                    MinMaxScaler(),
                )
            case "timestamp without time zone":
                return MinMaxScaler()
            case "date":
                return MinMaxScaler()
            case _:
                raise TypeError(
                    f"Missing encoder for dtype '{self.dtype}' on column {self.table}.{self.colname}"
                )

    def _prepare_values(self, values: Sequence) -> Sequence:
        match self.dtype:
            case "integer" | "smallint" | "bigint":
                return self._replace_nulls(values)
            case "character varying" | "text":
                values = self._replace_nulls(values)
                return pd.Series(
                    values
                )  # this allows for efficient conversion to ndarray
            case "timestamp without time zone":
                values = [v.timestamp() if v is not None else None for v in values]
                return self._replace_nulls(values)
            case "date":
                values = [
                    datetime.combine(v, time()).timestamp() if v is not None else None
                    for v in values
                ]
                return self._replace_nulls(values)
            case _:
                raise TypeError(
                    f"Missing encoder for dtype '{self.dtype}' on column {self.table}.{self.colname}"
                )

    def __repr__(self) -> str:
        return f"ColumnEncoder(colname='{self.colname}', table='{self.table}', dtype='{self.dtype}')"

    def __str__(self) -> str:
        prefix = self.__repr__()
        if self._encoder is None:
            return f"{prefix} [not fitted]"

        n_vals = len(self._values)
        return f"{prefix} [fitted on {n_vals} values]"
