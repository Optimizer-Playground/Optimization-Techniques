from __future__ import annotations

import collections
import json
import queue
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import postbound as pb
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder


@dataclass
class BinarizedQep:
    is_dummy: bool
    node: str
    outer_child: Optional[BinarizedQep]
    inner_child: Optional[BinarizedQep]
    cardinality: int
    cost: float
    cache_pct: float

    @staticmethod
    def create_for(
        plan: pb.QueryPlan, *, database: Optional[pb.Database] = None
    ) -> BinarizedQep:
        if plan.is_scan():
            cache_pct = _determine_cache_pct(plan, database)
            return BinarizedQep.scan(
                plan.node_type,
                cardinality=int(plan.estimated_cardinality),
                cost=plan.estimated_cost,
                cache_pct=cache_pct,
            )

        elif plan.is_join():
            assert plan.outer_child and plan.inner_child
            binarized_outer = BinarizedQep.create_for(
                plan.outer_child, database=database
            )
            binarized_inner = BinarizedQep.create_for(
                plan.inner_child, database=database
            )
            return BinarizedQep.pseudo_join(
                plan.node_type,
                binarized_outer,
                binarized_inner,
                cardinality=int(plan.estimated_cardinality),
                cost=plan.estimated_cost,
                cache_pct=0,
            )

        assert plan.input_node

        dummy_child = BinarizedQep.dummy()
        binarized_input = BinarizedQep.create_for(plan.input_node, database=database)
        return BinarizedQep.pseudo_join(
            plan.node_type,
            binarized_input,
            dummy_child,
            cardinality=int(plan.estimated_cardinality),
            cost=plan.estimated_cost,
            cache_pct=0,
        )

    @staticmethod
    def scan(
        node: str, *, cardinality: int, cost: float, cache_pct: float
    ) -> BinarizedQep:
        return BinarizedQep(False, node, None, None, cardinality, cost, cache_pct)

    @staticmethod
    def pseudo_join(
        node: str,
        outer_child: BinarizedQep,
        inner_child: BinarizedQep,
        *,
        cardinality: int,
        cost: float,
        cache_pct: float,
    ) -> BinarizedQep:
        return BinarizedQep(False, node, outer_child, inner_child, cardinality, cost, 0)

    @staticmethod
    def dummy() -> BinarizedQep:
        return BinarizedQep(True, "", None, None, -1, -1, 0)

    def outer(self) -> Optional[BinarizedQep]:
        # we need this method for the TCNN's flatten() method
        # functionally it is completely redundant.
        return self.outer_child

    def inner(self) -> Optional[BinarizedQep]:
        # we need this method for the TCNN's flatten() method
        # functionally it is completely redundant
        return self.inner_child

    def is_scan(self) -> bool:
        return not self.is_dummy and self.outer_child is None

    def is_join(self) -> bool:
        return (
            self.outer_child is not None
            and self.inner_child is not None
            and not self.inner_child.is_dummy
        )

    def is_intermediate(self) -> bool:
        return self.inner_child is not None and self.inner_child.is_dummy


class NodeType(IntEnum):
    Join = 1
    Scan = 2
    Intermediate = 3
    Dummy = 4


FeaturizedNode = collections.namedtuple(
    "FeaturizedNode", ["node_type", "encoding", "outer_child", "inner_child"]
)


def node_features(node: FeaturizedNode) -> np.ndarray:
    return node.encoding


def outer_child(node: FeaturizedNode) -> Optional[FeaturizedNode]:
    if node.node_type == NodeType.Dummy or node.node_type == NodeType.Scan:
        return None
    return node.outer_child


def inner_child(node: FeaturizedNode) -> Optional[FeaturizedNode]:
    if node.node_type == NodeType.Dummy or node.node_type == NodeType.Scan:
        return None
    return node.inner_child


def _determine_cache_pct(
    node: pb.QueryPlan, database: Optional[pb.Database] = None
) -> float:
    measures = node.measures
    if measures.cache_hits is not None and measures.cache_misses is not None:
        total_cache_accesses = measures.cache_hits + measures.cache_misses
        if total_cache_accesses == 0:
            return 1

        return measures.cache_hits / total_cache_accesses

    if database is None or not isinstance(database, pb.postgres.PostgresInterface):
        return 0

    assert node.base_table
    match node.operator:
        case pb.ScanOperator.SequentialScan:
            rels = [node.base_table.full_name]
        case pb.ScanOperator.IndexScan | pb.ScanOperator.IndexOnlyScan:
            rels = [node.params.index, node.base_table.full_name]
        case pb.ScanOperator.BitmapScan:
            rel_set = set()
            for child in node:
                if child.base_table is not None:
                    rel_set.add(child.base_table.full_name)
                if child.params.index:
                    rel_set.add(child.params.index)
            rels = list(rel_set)
        case _:
            raise ValueError(f"Unsupported scan operator: {node.operator}")

    pct_sum = 0
    stats = database.statistics()
    for rel in rels:
        total_pages = stats.n_pages(rel)
        buffered_pages = stats.n_buffered(rel)
        pct_sum += buffered_pages / total_pages

    return pct_sum / len(rels)


_PGNodeMap = {
    pb.ScanOperator.SequentialScan: "Seq Scan",
    pb.ScanOperator.IndexScan.value: "Index Scan",
    pb.ScanOperator.IndexOnlyScan.value: "Index Only Scan",
    pb.ScanOperator.BitmapScan: "Bitmap Heap Scan",
    pb.JoinOperator.NestedLoopJoin: "Nested Loop",
    pb.JoinOperator.HashJoin: "Hash Join",
    pb.JoinOperator.SortMergeJoin: "Merge Join",
    pb.IntermediateOperator.Materialize: "Materialize",
    pb.IntermediateOperator.Memoize: "Memoize",
    pb.IntermediateOperator.Sort.value: "Sort",
}


class BaoFeaturizer:
    @staticmethod
    def online(
        database: pb.Database, *, max_runtime_ms: float = 1000 * 60 * 60
    ) -> BaoFeaturizer:
        operators = pb.postgres.PostgresExplainNode.all_node_types()

        top_tables = queue.PriorityQueue(maxsize=3)
        for tab in database.schema().tables():
            card = database.statistics().total_rows(tab)
            if card is None:
                continue

            if not top_tables.full():
                top_tables.put((card, tab))
                continue

            lowest_card, lowest_tab = top_tables.get()
            if lowest_card > card:
                top_tables.put((lowest_card, lowest_tab))
            else:
                top_tables.put((card, tab))

        largest_tables: list[pb.TableReference] = []
        largest_cards: list[int] = []
        while not top_tables.empty():
            current_card, current_tab = top_tables.get()
            largest_cards.append(current_card)
            largest_tables.append(current_tab)

        max_card = int(np.prod(largest_cards))

        select_clause = pb.qal.Select.count_star()
        from_clause = pb.qal.From.create_for(largest_tables)
        cross_product = pb.qal.SqlQuery(
            select_clause=select_clause, from_clause=from_clause
        )

        max_cost = database.optimizer().cost_estimate(cross_product)

        return BaoFeaturizer(
            allowed_ops=operators,
            min_card=0,
            max_card=max_card,
            min_cost=0,
            max_cost=max_cost,
            min_runtime_ms=1,
            max_runtime_ms=max_runtime_ms,
        )

    @staticmethod
    def infer_from_sample(
        sample: pd.DataFrame, *, plan_col: str = "query_plan"
    ) -> BaoFeaturizer:
        allowed_ops: set[str] = set()
        min_card, max_card = np.inf, 0
        min_cost, max_cost = np.inf, 0
        min_runtime, max_runtime = np.inf, 0

        for plan in sample[plan_col]:
            assert isinstance(plan, pb.QueryPlan)
            for node in plan:
                allowed_ops.add(node.node_type)
                min_card = min(min_card, node.estimated_cardinality)
                max_card = max(max_card, node.estimated_cardinality)
                min_cost = min(min_cost, node.estimated_cost)
                max_cost = max(max_cost, node.estimated_cost)
                min_runtime = min(min_runtime, node.execution_time)
                max_runtime = max(max_runtime, node.execution_time)

        min_card = int(min_card)
        max_card = int(max_card)
        return BaoFeaturizer(
            allowed_ops=allowed_ops,
            min_card=min_card,
            max_card=max_card,
            min_cost=min_cost,
            max_cost=max_cost,
            min_runtime_ms=min_runtime,
            max_runtime_ms=max_runtime,
        )

    @staticmethod
    def pre_built(archive: Path | str) -> BaoFeaturizer:
        with open(archive, "r") as f:
            catalog = json.load(f)

        return BaoFeaturizer(
            allowed_ops=catalog["allowed_ops"],
            min_card=catalog["min_card"],
            max_card=catalog["max_card"],
            min_cost=catalog["min_cost"],
            max_cost=catalog["max_cost"],
            min_runtime_ms=catalog["min_runtime_ms"],
            max_runtime_ms=catalog["max_runtime_ms"],
        )

    @staticmethod
    def load_or_build(archive: Path | str, *, database: pb.Database) -> BaoFeaturizer:
        archive = Path(archive)
        if archive.is_file():
            return BaoFeaturizer.pre_built(archive)

        return BaoFeaturizer.online(database)

    def __init__(
        self,
        *,
        allowed_ops: Iterable[str],
        min_card: int,
        max_card: int,
        min_cost: float,
        max_cost: float,
        min_runtime_ms: float,
        max_runtime_ms: float,
    ) -> None:
        allowed_ops = list(allowed_ops)
        if "__bao_null__" not in allowed_ops:
            allowed_ops.append("__bao_null__")

        self._allowed_ops = allowed_ops
        self._op_enc = OneHotEncoder(sparse_output=False)
        self._op_enc.fit([(op,) for op in self._allowed_ops])

        self._min_card = min_card
        self._max_card = max_card
        self._min_cost = min_cost
        self._max_cost = max_cost
        self._min_runtime = min_runtime_ms
        self._max_runtime = max_runtime_ms

        self._runtime_pipeline = make_pipeline(
            FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
            MinMaxScaler((self._min_runtime, self._max_runtime)),
        )
        self._runtime_pipeline.fit(
            np.asarray([self._min_runtime, self._max_runtime]).reshape(-1, 1)
        )

    @property
    def out_shape(self) -> int:
        return (
            len(self._op_enc.categories_[0])
            + 1  # card enc
            + 1  # cost enc
            + 1  # cache_pct enc
        )

    def encode(self, node: BinarizedQep) -> np.ndarray:
        op_enc = self.encode_operator(node)
        card_enc = self.encode_cardinality(node)
        cost_enc = self.encode_cost(node)
        cache_pct_enc = self.encode_cache_pct(node)
        return np.concat(
            [
                op_enc,
                card_enc,
                cost_enc,
                cache_pct_enc,
            ],
        )

    def encode_plan(
        self,
        plan: pb.QueryPlan,
        *,
        database: Optional[pb.Database] = None,
    ) -> FeaturizedNode | tuple:
        binarized = BinarizedQep.create_for(plan, database=database)
        return self._featurize_qep(binarized)

    def encode_operator(self, node: BinarizedQep) -> np.ndarray:
        operator = "__bao_null__" if node.is_dummy else node.node
        operator = _PGNodeMap.get(operator, operator)
        enc = self._op_enc.transform([(operator,)])
        return enc[0]

    def encode_cardinality(self, node: BinarizedQep) -> np.ndarray:
        if node.is_dummy:
            return np.zeros(1)
        scaled = np.log(node.cardinality + 1) - self._min_card
        scaled /= self._max_card - self._min_card
        return np.asarray([scaled])

    def encode_cost(self, node: BinarizedQep) -> np.ndarray:
        if node.is_dummy:
            return np.zeros(1)
        scaled = np.log(node.cost + 1) - self._min_cost
        scaled /= self._max_cost - self._min_cost
        return np.asarray([scaled])

    def encode_cache_pct(self, node: BinarizedQep) -> np.ndarray:
        if node.is_dummy:
            return np.zeros(1)
        return np.asarray([node.cache_pct])

    def transform_runtime(self, runtime) -> np.ndarray:
        return self._runtime_pipeline.transform(runtime)

    def inverse_transform_runtime(self, scaled):
        return self._runtime_pipeline.inverse_transform(scaled)

    def store(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialized = {
            "allowed_ops": self._allowed_ops,
            "min_card": self._min_card,
            "max_card": self._max_card,
            "min_cost": self._min_cost,
            "max_cost": self._max_cost,
            "max_runtime_ms": self._max_runtime,
            "min_runtime_ms": self._min_runtime,
        }
        with open(path, "w") as f:
            json.dump(serialized, f)

    def _featurize_qep(self, node: BinarizedQep | None) -> FeaturizedNode:
        if node is None or node.is_dummy:
            encoding = torch.zeros(self.out_shape)
            return FeaturizedNode(NodeType.Dummy, encoding, [], [])

        encoding = self.encode(node)
        if node.is_scan():
            return FeaturizedNode(NodeType.Scan, encoding, [], [])
        elif node.is_join():
            return FeaturizedNode(
                NodeType.Join,
                encoding,
                self._featurize_qep(node.outer_child),
                self._featurize_qep(node.inner_child),
            )
        else:
            assert node.is_intermediate()
            return FeaturizedNode(
                NodeType.Intermediate,
                encoding,
                self._featurize_qep(node.outer_child),
                self._featurize_qep(node.inner_child),
            )
