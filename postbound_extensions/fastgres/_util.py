import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from bidict import bidict
from postbound import PhysicalOperator, ScanOperator, JoinOperator, PhysicalOperatorAssignment
from postbound.qal import LogicalOperator

from .hinting import HintSetFactory, HintSet, CORE_HINT_LIBRARY, HintLibrary
from tqdm import tqdm


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
        to_encode: int | float | datetime,
        min_value: int | float | datetime,
        max_value: int | float | datetime,
        offset: int | float | timedelta,
        round_values: int = 4
) -> int | float:
    return round((to_encode + offset - min_value) / (max_value - min_value + offset), round_values)


def save_json(to_save: Any, path: Path):
    json_dict = json.dumps(to_save)
    with path.open("w") as f:
        f.write(json_dict)


def load_json(path: Path) -> dict:
    with path.open() as f:
        loaded = json.load(f)
    return loaded


class FgPbConverter:

    pb_fg_log_map = bidict({
            LogicalOperator.Equal: "eq",
            LogicalOperator.NotEqual: "neq",
            LogicalOperator.Less: "lt",
            LogicalOperator.LessEqual: "lte",
            LogicalOperator.Greater: "gt",
            LogicalOperator.GreaterEqual: "gte",
            LogicalOperator.Like: "like",
            LogicalOperator.NotLike: "not_like",
            LogicalOperator.In: "in",
            LogicalOperator.Exists: "exists",
            LogicalOperator.Is: "is",
            LogicalOperator.IsNot: "is_not",
            LogicalOperator.Between: "between",
        })

    pb_fg_phy_map = bidict({
        ScanOperator.IndexOnlyScan: "enable_indexonlyscan",
        ScanOperator.SequentialScan: "enable_seqscan",
        ScanOperator.IndexScan: "enable_indexscan",
        JoinOperator.NestedLoopJoin: "enable_nestloop",
        JoinOperator.SortMergeJoin: "enable_mergejoin",
        JoinOperator.HashJoin: "enable_hashjoin",
    })

    operator_map = bidict({
        LogicalOperator.Equal: (0, 0, 1),
        LogicalOperator.NotEqual: (1, 1, 0),
        LogicalOperator.Less: (1, 0, 0),
        LogicalOperator.LessEqual: (1, 0, 1),
        LogicalOperator.Greater: (0, 1, 0),
        LogicalOperator.GreaterEqual: (0, 1, 1),
        LogicalOperator.Like: (1, 1, 1),
        LogicalOperator.NotLike: (0, 0, 0),
    })

    def fg_to_pb_log_operator(self, fg_operator: str) -> LogicalOperator:
        return self.pb_fg_log_map.inverse[fg_operator]

    def pb_to_fg_log_operator(self, pb_operator: LogicalOperator) -> str:
        return self.pb_fg_log_map[pb_operator]

    def fg_to_pb_phy_operator(self, fg_operator: str) -> ScanOperator | JoinOperator:
        return self.pb_fg_phy_map.inverse[fg_operator]

    def pb_to_fg_phy_operator(self, pb_operator: PhysicalOperator) -> str:
        return self.pb_fg_phy_map[pb_operator]

    def fg_to_pb_expr_operator(self, fg_operator: tuple[int, int, int]) -> LogicalOperator:
        return self.operator_map.inverse[fg_operator]

    def pb_to_fg_expr_operator(self, pb_operator: LogicalOperator) -> tuple[int, int, int]:
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
            hint_library: Optional[HintLibrary] = CORE_HINT_LIBRARY
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
