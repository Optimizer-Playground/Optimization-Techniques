import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from postbound import (
    JoinOperator,
    PhysicalOperator,
    PhysicalOperatorAssignment,
    ScanOperator,
)
from postbound.qal import BinaryOperator, UnaryOperator
from tqdm import tqdm

from ._mapping import BidirectionalMap
from .hinting import CORE_HINT_LIBRARY, HintLibrary, HintSet, HintSetFactory


def prepare_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(f"Directory already exists and is not empty: {path}")
        return

    path.mkdir(parents=True, exist_ok=False)
    tqdm.write(f"Created directory: {path}")


def min_max_encode(
        to_encode: float | Decimal | date,
        min_value: float | Decimal | date,
        max_value: float | Decimal | date,
        offset: float | timedelta,
        round_values: int = 4
) -> float:
    if isinstance(offset, timedelta):
        if not isinstance(to_encode, date) or not isinstance(min_value, date) or not isinstance(max_value, date):
            raise TypeError("A timedelta offset requires date or datetime values")
        value = to_encode if isinstance(to_encode, datetime) else datetime.combine(to_encode, time.min)
        lower = min_value if isinstance(min_value, datetime) else datetime.combine(min_value, time.min)
        upper = max_value if isinstance(max_value, datetime) else datetime.combine(max_value, time.min)
        encoded = (value + offset - lower) / (upper - lower + offset)
    else:
        if isinstance(to_encode, date) or isinstance(min_value, date) or isinstance(max_value, date):
            raise TypeError("Date or datetime values require a timedelta offset")
        encoded = (float(to_encode) + offset - float(min_value)) / (float(max_value) - float(min_value) + offset)
    return round(encoded, round_values)


def save_json(to_save: Any, path: Path):
    json_dict = json.dumps(to_save)
    with path.open("w") as f:
        f.write(json_dict)


def load_json(path: Path) -> dict:
    with path.open() as f:
        loaded = json.load(f)
    return loaded


class FgPbConverter:

    pb_fg_log_map = BidirectionalMap({
            BinaryOperator.Equal: "eq",
            BinaryOperator.NotEqual: "neq",
            BinaryOperator.Less: "lt",
            BinaryOperator.LessEqual: "lte",
            BinaryOperator.Greater: "gt",
            BinaryOperator.GreaterEqual: "gte",
            BinaryOperator.Like: "like",
            BinaryOperator.NotLike: "not_like",
            BinaryOperator.ILike: "ilike",
            BinaryOperator.NotILike: "not_ilike",
            BinaryOperator.In: "in",
            BinaryOperator.NotIn: "not_in",
            UnaryOperator.Exists: "exists",
            UnaryOperator.IsNull: "is_null",
            UnaryOperator.IsNotNull: "is_not_null",
            UnaryOperator.IsTrue: "is_true",
            UnaryOperator.IsNotTrue: "is_not_true",
            UnaryOperator.IsFalse: "is_false",
            UnaryOperator.IsNotFalse: "is_not_false",
            BinaryOperator.Between: "between",
        })

    pb_fg_phy_map = BidirectionalMap({
        ScanOperator.IndexOnlyScan: "enable_indexonlyscan",
        ScanOperator.SequentialScan: "enable_seqscan",
        ScanOperator.IndexScan: "enable_indexscan",
        JoinOperator.NestedLoopJoin: "enable_nestloop",
        JoinOperator.SortMergeJoin: "enable_mergejoin",
        JoinOperator.HashJoin: "enable_hashjoin",
    })

    operator_map = BidirectionalMap({
        BinaryOperator.Equal: (0, 0, 1),
        BinaryOperator.NotEqual: (1, 1, 0),
        BinaryOperator.Less: (1, 0, 0),
        BinaryOperator.LessEqual: (1, 0, 1),
        BinaryOperator.Greater: (0, 1, 0),
        BinaryOperator.GreaterEqual: (0, 1, 1),
        BinaryOperator.Like: (1, 1, 1),
        BinaryOperator.NotLike: (0, 0, 0),
    })

    def fg_to_pb_log_operator(self, fg_operator: str) -> BinaryOperator | UnaryOperator:
        return self.pb_fg_log_map.inverse[fg_operator]

    def pb_to_fg_log_operator(self, pb_operator: BinaryOperator | UnaryOperator) -> str:
        return self.pb_fg_log_map[pb_operator]

    def fg_to_pb_phy_operator(self, fg_operator: str) -> ScanOperator | JoinOperator:
        return self.pb_fg_phy_map.inverse[fg_operator]

    def pb_to_fg_phy_operator(self, pb_operator: ScanOperator | JoinOperator) -> str:
        return self.pb_fg_phy_map[pb_operator]

    def fg_to_pb_expr_operator(self, fg_operator: tuple[int, int, int]) -> BinaryOperator:
        return self.operator_map.inverse[fg_operator]

    def pb_to_fg_expr_operator(self, pb_operator: BinaryOperator) -> tuple[int, int, int]:
        return self.operator_map[pb_operator]

    def fg_to_pb_hint_set(self, fg_hs: HintSet) -> PhysicalOperatorAssignment:
        fg_lib = fg_hs.library
        if fg_lib != CORE_HINT_LIBRARY:
            raise ValueError("Trying to convert non-core hints to PostBound. Currently, six core hints are supported.")

        pb_phy_assignment = PhysicalOperatorAssignment()
        for db_instruction, hs_value in fg_hs:
            pb_phy_assignment.set_operator_enabled_globally(
                operator=self.fg_to_pb_phy_operator(db_instruction),
                enabled=hs_value,
            )

        return pb_phy_assignment

    def pb_to_fg_hint_set(
            self,
            pb_hs: PhysicalOperatorAssignment,
            hint_library: HintLibrary = CORE_HINT_LIBRARY
    ) -> HintSet:

        db_instructions = [hint.db_instr for hint in hint_library.hint_list]
        hsf = HintSetFactory(hint_library)

        globally_enabled: frozenset[PhysicalOperator] = pb_hs.get_globally_enabled_operators()

        disabled_instructions = set()
        for db_instruction in db_instructions:
            phy_op: PhysicalOperator = self.fg_to_pb_phy_operator(db_instruction)
            if phy_op not in globally_enabled:
                disabled_instructions.add(db_instruction)

        return hsf.hint_set_from_disabled(disabled_instructions)
