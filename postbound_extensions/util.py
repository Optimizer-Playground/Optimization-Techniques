from __future__ import annotations

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
