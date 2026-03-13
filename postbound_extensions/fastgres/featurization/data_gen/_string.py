
from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass, field

from .._dbutil import DatabaseConnection
from ..._util import save_json, load_json, min_max_encode
from ._statistics import StatisticsComponent

import hashlib


class FastgresStringComponent(StatisticsComponent):
    skipped_string_columns = {
        "account": ["display_name"],
        "answer": ["title", "body"],
        "question": ["title", "tagstring", "body"],
        "site": ["site_name"],
        "tag": ["name"],
        "badge": ["name"],
        "comment": ["body"]
    }

    @dataclass
    class FastgresLabelEncoder:
        _classes: list[str] = field(default_factory=list)
        _encoder: dict[str, int] = field(default_factory=dict)

        def fit_encoder(self, y: list[str], sorty_by: list[int]) -> FastgresStringComponent.FastgresLabelEncoder:
            if len(y) != len(sorty_by):
                raise ValueError("Length of y and sorty_by must be the same.")

            counts = defaultdict(int)
            for key, count in zip(y, sorty_by):
                counts[str(key)] += count
            sorted_keys = sorted(counts.keys(), key=lambda x: counts[x])

            self._classes = sorted_keys
            self._encoder = {key: idx for idx, key in enumerate(sorted_keys)}
            return self

        def transform(self, values: str | list[str]) -> list[int]:
            if isinstance(values, str):
                values = [values]
            offset = 1
            round_values = 4
            min_v = 0
            max_v = max(self._encoder.values())
            encoded_values = [self._encoder.get(item, -1) for item in values]
            normalized_values = [min_max_encode(value, min_v, max_v, offset, round_values) for value in encoded_values]
            # return [self._encoder.get(item, -1) for item in values]  # Returns -1 if item not found
            return normalized_values

        def to_dict(self) -> dict[str, Any]:
            return {
                "classes_": self._classes,
                "encoder": self._encoder
            }

        @classmethod
        def from_dict(cls, encoder_dict: dict[str, Any]) -> FastgresStringComponent.FastgresLabelEncoder:
            obj = FastgresStringComponent.FastgresLabelEncoder()
            obj._classes = encoder_dict["classes_"]
            obj._encoder = encoder_dict["encoder"]
            return obj

    def __init__(self):
        self._encoders = {}

    @property
    def available_encoders(self) -> dict[str, list[str]]:
        return {table: list(cols.keys()) for table, cols in self._encoders.items()}

    def build(self, dbc: DatabaseConnection, **kwargs) -> None:

        def _should_skip(conn: DatabaseConnection, t: str, c: str) -> bool:
            return (
                    "stack" in conn.pbc.connect_string or "stack_overflow" in conn.pbc.connect_string
                    and t in self.skipped_string_columns
                    and c in self.skipped_string_columns[t]
            )

        valid_types = {"character varying", "character", "text"}
        tables_and_columns = dbc.filter_by_types(valid_types)

        for column_ref in tqdm(tables_and_columns, desc="Building Label Encoder Statistics"):
            t_n = column_ref.table.full_name
            c_n = column_ref.name
            if _should_skip(dbc, t_n, c_n):
                continue

            columns_and_counts = dbc.column_count(column_ref)
            if not columns_and_counts:
                tqdm.write(f"No data found for table: {t_n}, column: {c_n}. Skipping encoder.")
                continue

            tqdm.write(f"Fitting label encoder to table: {t_n}, column: {c_n}")
            y, sorty_by = zip(*columns_and_counts)
            label_encoder = self.FastgresLabelEncoder().fit_encoder(list(y), list(sorty_by))
            self._encoders.setdefault(t_n, {})[c_n] = label_encoder

    def to_dict(self) -> dict:
        return {
            table: {col: enc.to_dict() for col, enc in columns.items()}
            for table, columns in self._encoders.items()
        }

    def save(self, dir_path: Path):
        return save_json(self.to_dict(), dir_path / "label_encoders.json")

    @classmethod
    def load(cls, dir_path: Path) -> FastgresStringComponent:
        encoders = load_json(dir_path / "label_encoders.json")
        return cls.from_dict(encoders)

    @classmethod
    def from_dict(cls, data: dict) -> FastgresStringComponent:
        obj = cls()
        obj._encoders = {
            table: {
                column: FastgresStringComponent.FastgresLabelEncoder.from_dict(encoder_dict)
                for column, encoder_dict in columns.items()
            }
            for table, columns in data.items()
        }
        return obj

    def transform(self, table: str, column: str, values: str | list[str], skipped: dict) -> list[int]:
        if values is None:
            return [0]
        if isinstance(values, str):
            values = [values]
        if self.is_skipped(table, column, skipped):
            return [self.hash_character(value) for value in values]
        return self._encoders[table][column].transform(values)

    @staticmethod
    def is_skipped(table: str, column: str, skipped_columns: dict):
        return table in skipped_columns and column in skipped_columns[table]["columns"]

    @staticmethod
    def hash_character(value: str):
        max_encoding_size: int = 2 ** 64  # md5 output is 64 bit standard
        min_encoding_size: int = 0  # should be min for hashes
        encoding_offset: int = 1
        round_values = 4
        b_string = bytes(value, "utf-8")
        hash_value = int.from_bytes(hashlib.md5(b_string).digest()[:8], 'little')
        return min_max_encode(hash_value, min_encoding_size, max_encoding_size, encoding_offset, round_values)
