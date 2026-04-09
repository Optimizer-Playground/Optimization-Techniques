"""Miscellaneous utility functions that are of use for multiple optimizers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import postbound as pb
from torch.utils.data import Dataset


def distinct_values_query(col: pb.ColumnReference) -> pb.SqlQuery:
    """Creates an SQL query to load all distinct values for a specific table.

    The values will be calculated in their natural ordering.
    """
    if (table := col.table) is None:
        raise ValueError("Column must be bound to a table")

    select_clause = pb.qal.Select.create_for(col, distinct=True)
    from_clause = pb.qal.From.create_for(table)
    orderby_clause = pb.qal.OrderBy.create_for(col)

    return pb.qal.ImplicitSqlQuery(
        select_clause=select_clause,
        from_clause=from_clause,
        orderby_clause=orderby_clause,
    )


def min_max_values_query(col: pb.ColumnReference) -> pb.SqlQuery:
    """Creates an SQL query to calculate the minimum and maximum value for a specific column."""
    if (table := col.table) is None:
        raise ValueError("Column must be bound to a table")

    min_col = pb.qal.FunctionExpression.create_min(col)
    max_col = pb.qal.FunctionExpression.create_max(col)
    select_clause = pb.qal.Select(
        [pb.qal.BaseProjection(min_col), pb.qal.BaseProjection(max_col)]
    )
    from_clause = pb.qal.From.create_for(table)

    return pb.qal.ImplicitSqlQuery(
        select_clause=select_clause,
        from_clause=from_clause,
    )


def wrap_logger(logger: bool | pb.util.Logger) -> pb.util.Logger:
    """Creates a logger instance.

    If the argument already is a proper PostBOUND logger, it will not be modified.
    A boolean is interpreted as a flag to indicate whether logging is enabled or not.
    """
    if isinstance(logger, bool):
        return pb.util.standard_logger(logger)
    return logger


def load_training_samples(
    path: pd.DataFrame | Path | str,
    *,
    query_col: str = "query",
    verbose: bool | pb.util.Logger = False,
) -> pd.DataFrame:
    """Loads a Pandas DataFrame of training samples.

    The samples can be provided in different formats (e.g. CSV or Parquet). Queries will be parsed
    into proper PostBOUND query objects if necessary.

    Notes
    ------
    The main purpose of this method is to allow broad flexibility in how users can provide training
    samples (e.g., byspecifying the file path or by directly providing the actual data) while
    keeping the optimizers from implementing the same parsing logic over and over again.
    """
    samples = pb.util.read_df(path) if isinstance(path, (Path, str)) else path
    if isinstance(samples[query_col].iloc[0], pb.SqlQuery):
        # To support older Pandas versions that might not support dtype=str yet,
        # we perform an isinstance() check to determine if parsing is necessary.
        return samples

    log = wrap_logger(verbose)
    log("Parsing queries")
    samples[query_col] = samples[query_col].map(pb.parse_query)
    return samples


class PandasDataset(Dataset):
    """Wraps a Pandas DataFrame into a PyTorch Dataset.

    Each row in the DataFrame is interpreted as a single training sample. All columns are included.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return list(self.df.iloc[idx])


def _noop_parser[T](value: T) -> T:
    return value


def _date_parser(d: str) -> date:
    return datetime.fromisoformat(d).date()


def make_json_parser(column_dtype: str) -> Callable[[Any], Any]:
    """Creates a parser function to load the JSON-serialized values of a column.

    This functions as the inverse to PostBOUND's ``to_json()` utilitiy.
    """

    match column_dtype:
        case "timestamp with time zone" | "timestamp without time zone":
            return datetime.fromisoformat
        case "date":
            return _date_parser
        case "varchar" | "text" | "integer":
            return _noop_parser
        case _:
            raise ValueError(f"Missing JSON parser for column type {column_dtype}")
