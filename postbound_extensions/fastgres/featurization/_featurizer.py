from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Sequence, Collection

import numpy as np
import postbound as pb
from postbound.qal import CompoundPredicate, AbstractPredicate, UnaryPredicate, InPredicate, BetweenPredicate, \
    BinaryPredicate, SimpleFilter, LogicalOperator
from tqdm.contrib.bells import tqdm

from ._dbutil import DatabaseConnection
from .data_gen import FastgresMinMaxComponent, FastgresStringComponent, FastgresWildcardComponent, StatisticsComponent
from .._util import load_json, prepare_dir
from ..context import ContextManager, Context


class ModuleError(Exception):
    """Exception raised for errors in a feature module."""
    pass


@dataclass
class EncodingInformation:

    min_max_dict: FastgresMinMaxComponent
    label_encoders: FastgresStringComponent
    wildcard_dict: FastgresWildcardComponent
    skipped_columns: dict

    @staticmethod
    def load_skipped_columns(path: Path):
        # temporary, manual workaround for columns high distinct value percentage
        skipped_path = path / "skipped.json"
        if not os.path.exists(skipped_path):
            tqdm.write("No skipped table columns found...")
            return dict()
        else:
            return load_json(skipped_path)

    @classmethod
    def load_encoding_info(cls, base_dir: Path):
        try:
            tqdm.write("Loading MinMax Components...")
            min_max_dict = FastgresMinMaxComponent.load(base_dir)
            tqdm.write("Loading String Components...")
            label_encoders = FastgresStringComponent.load(base_dir)
            tqdm.write("Loading Wildcard Components...")
            wildcard_dict = FastgresWildcardComponent.load(base_dir)
            tqdm.write("Loading Skipped Components...")
            skipped_columns = cls.load_skipped_columns(base_dir)
            tqdm.write("...done.")
        except ValueError:
            raise ValueError("Exception loading dictionaries")

        return cls(min_max_dict, label_encoders, wildcard_dict, skipped_columns)

    @classmethod
    def from_dicts(cls, components: list[StatisticsComponent], skipped_columns: dict) -> EncodingInformation:
        min_max_dict = None
        label_encoders = None
        wildcard_dict = None
        for comp in components:
            match comp:
                case FastgresMinMaxComponent():
                    min_max_dict = comp
                case FastgresStringComponent():
                    label_encoders = comp
                case FastgresWildcardComponent():
                    wildcard_dict = comp
        if min_max_dict is None or label_encoders is None or wildcard_dict is None:
            raise ValueError(f"Insufficient statistic component provided: {components}")
        return cls(min_max_dict, label_encoders, wildcard_dict, skipped_columns)

    @staticmethod
    def build(dbc: DatabaseConnection, workload: pb.Workload, skipped_columns: dict) -> EncodingInformation:
        modules = {
            "min_max": FastgresMinMaxComponent(),
            "string": FastgresStringComponent(),
            "wildcard": FastgresWildcardComponent(),
        }
        [module.build(dbc=dbc, workload=workload) for module in tqdm(modules.values(), desc="Building FG Module Statistic")]
        return EncodingInformation.from_dicts(list(modules.values()), skipped_columns=skipped_columns)

    @staticmethod
    def build_and_save(dbc: DatabaseConnection, workload: pb.Workload, skipped_columns: dict, *, save_dir: Path | str) -> EncodingInformation:
        enc = EncodingInformation.build(dbc, workload, skipped_columns)
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        prepare_dir(save_dir)
        [module.save(save_dir) for module in [enc.min_max_dict, enc.label_encoders, enc.wildcard_dict]]
        return enc


class ColumnType(Enum):
    CHARACTER_VARYING = "character varying"
    TEXT = "text"
    INTEGER = "integer"
    TIMESTAMP_WITHOUT_TIMEZONE = "timestamp without time zone"
    DATE = "date"
    NUMERIC = "numeric"
    UNHANDLED = "unhandled"

    @classmethod
    def from_string(cls, s: str) -> ColumnType:
        for member in cls:
            if member is not cls.UNHANDLED and member.value == s:
                return member
        return cls.UNHANDLED


class FastgresPredicatesCollector(
    pb.qal.PredicateVisitor[Collection[pb.qal.SimpleFilter]]
):

    def visit_binary_predicate(
            self, predicate: BinaryPredicate, *args, **kwargs
    ) -> Collection[SimpleFilter]:
        if SimpleFilter.can_wrap(predicate) and predicate.is_filter():
            return [SimpleFilter.wrap(predicate)]
        return []

    def visit_between_predicate(
            self, predicate: BetweenPredicate, *args, **kwargs
    ) -> Collection[SimpleFilter]:
        return []

    def visit_in_predicate(
            self, predicate: InPredicate, *args, **kwargs
    ) -> Collection[SimpleFilter]:
        if SimpleFilter.can_wrap(predicate) and predicate.is_filter():
            return [SimpleFilter.wrap(predicate)]
        return []

    def visit_unary_predicate(
            self, predicate: UnaryPredicate, *args, **kwargs
    ) -> Collection[SimpleFilter]:
        return []

    def visit_not_predicate(
            self, predicate: CompoundPredicate, child_predicate: AbstractPredicate, *args, **kwargs
    ) -> Collection[SimpleFilter]:
        return []

    def visit_or_predicate(
            self, predicate: CompoundPredicate, components: Sequence[AbstractPredicate], *args, **kwargs
    ) -> Collection[SimpleFilter]:
        return []

    def visit_and_predicate(
            self, predicate: CompoundPredicate, components: Sequence[AbstractPredicate], *args, **kwargs
    ) -> Collection[SimpleFilter]:
        return pb.util.flatten(pred.accept_visitor(self) for pred in components)


class FastgresFeaturization:
    surjective_operator_map = {
        LogicalOperator.Equal: (0., 0., 1.),
        LogicalOperator.NotEqual: (1., 1., 0.),
        LogicalOperator.Less: (1., 0., 0.),
        LogicalOperator.LessEqual: (1., 0., 1.),
        LogicalOperator.Greater: (0., 1., 0.),
        LogicalOperator.GreaterEqual: (0., 1., 1.),
        LogicalOperator.Like: (1., 1., 1.),
        LogicalOperator.NotLike: (1., 1., 0.),

        LogicalOperator.ILike: (1., 1., 1.),  # Workaround for current FG implementation
        LogicalOperator.NotILike: (1., 1., 0.),

        LogicalOperator.In: (0., 0., 1.),
        LogicalOperator.Exists: (0., 0., 0.),

        LogicalOperator.Is: (0., 0., 1.),  # Currently unsupported
        LogicalOperator.IsNot: (1., 1., 0.),

        LogicalOperator.Between: (0., 0., 1.)
    }

    @dataclass
    class Components:
        schema: dict[str, dict[str, str]]  # table, column, d_type
        component_dict: dict = field(default_factory=dict)

        def add_entry(self, table: str, column: str, feature: tuple[float, float, float, float]):
            self.component_dict.setdefault(table, {}).setdefault(column, []).append(feature)

        def get_component(self, table: str, column: str) -> tuple[float, float, float, float]:
            try:
                return self.component_dict[table][column][-1]
            except KeyError:
                return 0.0, 0.0, 0.0, 0.0

        def __iter__(self):
            for table in sorted(self.schema.keys()):
                for column in sorted(self.schema[table].keys()):
                    yield table, column

        def arrange(self) -> list[float]:
            return_list: list[float] = list()
            for table, col in self:
                return_list.extend(self.get_component(table, col))
            return return_list

    def __init__(
            self,
            dbc: DatabaseConnection | pb.postgres.PostgresInterface,
            context: ContextManager,
            *,
            enc_info: EncodingInformation = None,
            enc_dir: Path | str = None,
    ):
        if isinstance(dbc, pb.postgres.PostgresInterface):
            dbc = DatabaseConnection(dbc)
        if enc_info is None and enc_dir is None:
            raise ValueError("Either enc_info or enc_dir must be provided")
        if enc_info is not None:
            self.statistics = enc_info
        else:
            if isinstance(enc_dir, str):
                enc_dir = Path(enc_dir)
            self.statistics = EncodingInformation.load_encoding_info(enc_dir)
        self.dbc = dbc
        self.ctx_manager = context

    def transform_single(self, query: pb.SqlQuery, **kwargs) -> np.ndarray:
        simple_predicates: Collection[SimpleFilter] = query.where_clause.predicate.accept_visitor(
            FastgresPredicatesCollector()
        )
        context: Context = self.ctx_manager[query]
        components = FastgresFeaturization.Components(context.schema.to_dict())
        for predicate in simple_predicates:
            table_name: str = predicate.column.table.full_name
            column_name: str = predicate.column.name
            operation: LogicalOperator = predicate.operation

            operator_feature: tuple[float, float, float] = FastgresFeaturization.surjective_operator_map[operation]
            filter_feature: float = self.featurize_filter(predicate)
            if isinstance(filter_feature, list):
                filter_feature = filter_feature[0]
            components.add_entry(table_name, column_name, operator_feature + (filter_feature,))
        return np.asarray(components.arrange(), dtype=np.float64)

    def transform(self, queries: list[pb.SqlQuery] | pb.SqlQuery, **kwargs) -> np.ndarray:
        if isinstance(queries, pb.SqlQuery):
            queries = [queries]
        return np.vstack([self.transform_single(q, **kwargs) for q in queries])

    def featurize_filter(self, predicate: SimpleFilter) -> float:
        table_name: str = predicate.column.table.full_name
        column_name: str = predicate.column.name
        value = predicate.value

        d_type: str = self.dbc.schema.datatype(predicate.column)
        col_type: ColumnType = ColumnType.from_string(d_type)
        match predicate._predicate:
            case BinaryPredicate(_, _, _):
                match col_type:
                    case ColumnType.CHARACTER_VARYING | ColumnType.TEXT:
                        if predicate.operation in (LogicalOperator.Like, LogicalOperator.NotLike,
                                                   LogicalOperator.ILike, LogicalOperator.NotILike):
                            return self.statistics.wildcard_dict.transform(table_name, column_name, value)
                        else:
                            return self.statistics.label_encoders.transform(
                                table_name, column_name, value, self.statistics.skipped_columns
                            )
                    case ColumnType.INTEGER | ColumnType.NUMERIC:
                        return self.statistics.min_max_dict.transform(table_name, column_name, value)
                    case ColumnType.TIMESTAMP_WITHOUT_TIMEZONE | ColumnType.DATE:
                        return self.statistics.min_max_dict.transform_time(table_name, column_name, value)
                    case ColumnType.UNHANDLED:
                        raise ModuleError(f"Unknown column type: {self.dbc.fg_schema.schema[table_name][column_name]}")
                    case _:
                        raise ModuleError(f"Unknown Column Type: {col_type}")
            case InPredicate(_, _):
                match col_type:
                    case ColumnType.CHARACTER_VARYING | ColumnType.TEXT:
                        vals = self.statistics.label_encoders.transform(
                            table_name, column_name, value, self.statistics.skipped_columns
                        )
                        return sum(vals) / len(vals)
                    case ColumnType.INTEGER | ColumnType.NUMERIC:
                        vals = [
                            self.statistics.min_max_dict.transform(table_name, column_name, value)
                            for value in value
                        ]
                        return sum(vals) / len(vals)
                    case _:
                        raise ModuleError(f"Unknown Column Type for IN-Filter: {col_type} -> {value}")
            case _:
                raise ModuleError(f"Trying to encode non-supported Predicate Types: {type(predicate)}, {predicate}")
