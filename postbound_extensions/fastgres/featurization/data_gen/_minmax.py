
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ._statistics import StatisticsComponent
from .._dbutil import DatabaseConnection
from ..._util import min_max_encode


class FastgresMinMaxComponent(StatisticsComponent):

    def __init__(self):
        self.dict = None

    def get(self, table: str, column: str) -> tuple[Any, Any]:
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

    def transform(self, table: str, column: str, to_encode: int | float) -> int | float:
        min_value, max_value = self.get(table, column)
        if min_value is None or max_value is None:
            return 1.0
        offset = 0.001
        round_values = 4
        return min_max_encode(to_encode, min_value, max_value, offset, round_values)

    def transform_time(self, table: str, column: str, to_encode: str) -> int | float:
        min_value, max_value = self.get(table, column)
        offset = timedelta(days=1)
        round_values = 4
        format_string = "%Y-%m-%d"
        try:
            filter_value = datetime.strptime(to_encode, format_string)
        except ValueError:
            raise ValueError(f"Invalid format: {to_encode}, {format_string}")
        return min_max_encode(filter_value, min_value, max_value, offset, round_values=round_values)

    class MinMaxEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            return super().default(obj)

    class MinMaxDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            super().__init__(object_hook=self.object_hook, *args, **kwargs)

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
