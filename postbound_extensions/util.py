from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import pandas as pd
import postbound as pb
from torch.utils.data import Dataset


def distinct_values_query(col: pb.ColumnReference) -> pb.SqlQuery:
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
    if (table := col.table) is None:
        raise ValueError("Column must be bound to a table")

    min_col = pb.qal.FunctionExpression.create_min(col)
    max_col = pb.qal.FunctionExpression.create_max(col)
    select_clause = pb.qal.Select.create_for([min_col, max_col])
    from_clause = pb.qal.From.create_for(table)

    return pb.qal.ImplicitSqlQuery(
        select_clause=select_clause,
        from_clause=from_clause,
    )


def wrap_logger(logger: bool | pb.util.Logger) -> pb.util.Logger:
    if isinstance(logger, bool):
        return pb.util.standard_logger(logger)
    return logger


class PandasDataset(Dataset):
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
    match column_dtype:
        case "timestamp with time zone" | "timestamp without time zone":
            return datetime.fromisoformat
        case "date":
            return _date_parser
        case "varchar" | "text" | "integer":
            return _noop_parser
        case _:
            raise ValueError(f"Missing JSON parser for column type {column_dtype}")
