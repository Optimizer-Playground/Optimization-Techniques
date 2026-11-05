
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..._util import min_max_encode
from .._dbutil import DatabaseConnection
from ._statistics import StatisticsComponent


class FastgresMinMaxComponent(StatisticsComponent):

    def __init__(self):
        self.dict: dict[str, dict[str, dict[str, Any]]] | None = None

    def get(self, table: str, column: str) -> tuple[Any, Any]:
        if self.dict is None:
            raise ValueError("Min/max statistics must be built or loaded first")
        return self.dict[table][column]["min"], self.dict[table][column]["max"]

    def build(self,  dbc: DatabaseConnection, **kwargs) -> dict[str, dict[str, dict]]:
        types = {'integer', 'timestamp without time zone', 'date', 'numeric'}
        filtered_schema = dbc.filter_by_types(types)
        self.dict = dict()
        for column_ref in tqdm(filtered_schema, desc="Building Min Max Dicts"):
            t_n = column_ref.table.full_name
            c_n = column_ref.name
            min_value, max_value = dbc.min_max(column_ref)
            self.dict.setdefault(t_n, {})[c_n] = {"min": min_value, "max": max_value}
        return self.dict

    def save(self, dir_path: Path):
        path = dir_path / "mm_dict.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.dict, f, cls=self.MinMaxEncoder)
        return path

    def to_dict(self):
        return self.dict

    @classmethod
    def load(cls, dir_path: Path) -> FastgresMinMaxComponent:
        obj = cls()
        path = dir_path / "mm_dict.json"
        with path.open("r", encoding="utf-8") as f:
            obj.dict = json.load(f, cls=obj.MinMaxDecoder)
        return obj

    def transform(self, table: str, column: str, to_encode: float) -> int | float:
        min_value, max_value = self.get(table, column)
        if min_value is None or max_value is None:
            return 1.0
        offset = 0.001
        round_values = 4
        return min_max_encode(to_encode, min_value, max_value, offset, round_values)

    def transform_time(self, table: str, column: str, to_encode: str | date) -> float:
        min_value, max_value = self.get(table, column)
        if min_value is None or max_value is None:
            return 1.0
        offset = timedelta(days=1)
        round_values = 4
        try:
            filter_value = datetime.fromisoformat(to_encode) if isinstance(to_encode, str) else to_encode
        except ValueError:
            raise ValueError(f"Invalid ISO date or timestamp: {to_encode}") from None
        return min_max_encode(filter_value, min_value, max_value, offset, round_values=round_values)

    class MinMaxEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            if isinstance(o, Decimal):
                return float(o)
            return super().default(o)

    class MinMaxDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, object_hook=self.object_hook, **kwargs)

        @staticmethod
        def object_hook(obj: dict[str, Any]) -> Any:
            for key, value in obj.items():
                if isinstance(value, str):
                    try:
                        if 'T' in value:
                            obj[key] = datetime.fromisoformat(value)
                        else:
                            obj[key] = date.fromisoformat(value)
                    except ValueError:
                        pass
            return obj
