"""
Bao Plan Featurization Logic
    Created as part of the Optimization Techniques Project

Copyright (C) 2026 Rico Bergmann

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
"""

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
    """A pre-processed query plan node that is suitable for featurization.

    A binarized node is guaranteed to have exactly zero (for base scans) or two children (for joins
    and other intermediate nodes). Intermediate nodes will typically have a dummy (i.e. all values 0)
    inner child. The node contains exactly the information that will be fed into the featurizer.

    Parameters
    ----------
    is_dummy: bool
        Whether this node is a dummy node. Dummy nodes are used to binarize intermediate nodes that
        would normally only have a single child.
    node: str
        The operator name of this node, e.g. "Seq Scan", "Hash Join", etc.
    outer_child: Optional[BinarizedQep]
        The outer child of this node. For scans, this will be None. For all other nodes, this will
        be a proper BinarizedQep node (i.e. not a dummy).
    inner_child: Optional[BinarizedQep]
        The inner child of this node. For scans, this will be None. For joins, this will be a proper
        BinarizedQep node (i.e. not a dummy). For intermediate nodes, this will be a dummy node.
    cardinality: int
        The estimated cardinality for this node. This is the raw estimate, not a scaled value.
    cost: float
        The estimated cost for this node. This is the raw estimate, not a scaled value.
    cache_pct: float
        The estimated cache percentage for this node.
    """

    is_dummy: bool
    node: str
    outer_child: Optional[BinarizedQep]
    inner_child: Optional[BinarizedQep]
    cardinality: int
    cost: float
    cache_pct: float

    @staticmethod
    def create_for(
        plan: pb.QueryPlan,
        *,
        cache_state: DatabaseCacheState,
    ) -> BinarizedQep:
        """Recursively transforms the plan node and all of its children into a binarized QEP."""
        if plan.is_scan():
            cache_pct = cache_state.determine_cache_pct(plan)
            return BinarizedQep.scan(
                plan.node_type,
                cardinality=int(plan.estimated_cardinality),
                cost=plan.estimated_cost,
                cache_pct=cache_pct,
            )

        elif plan.is_join():
            assert plan.outer_child and plan.inner_child
            binarized_outer = BinarizedQep.create_for(
                plan.outer_child, cache_state=cache_state
            )
            binarized_inner = BinarizedQep.create_for(
                plan.inner_child, cache_state=cache_state
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
        binarized_input = BinarizedQep.create_for(
            plan.input_node, cache_state=cache_state
        )
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
        """Transforms the scan node into its binarized equivalent."""
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
        """Creates a binarized node for the given join."""
        return BinarizedQep(False, node, outer_child, inner_child, cardinality, cost, 0)

    @staticmethod
    def dummy() -> BinarizedQep:
        """Creates a dummy binarized node."""
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


class DatabaseCacheState:
    def __init__(self, database: pb.Database) -> None:
        self._db = database
        self._stats = self._db.statistics()

        if isinstance(self._stats, pb.postgres.PostgresStatisticsInterface):
            self._cache_state = self._stats.buffer_state()
        else:
            self._cache_state = {}

    def determine_cache_pct(self, node: pb.QueryPlan) -> float:
        measures = node.measures

        if measures.cache_hits is not None and measures.cache_misses is not None:
            total_cache_accesses = measures.cache_hits + measures.cache_misses
            if total_cache_accesses == 0:
                return 1

            return measures.cache_hits / total_cache_accesses

        if not isinstance(self._db, pb.postgres.PostgresInterface):
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
        for rel in rels:
            total_pages = self._stats.n_pages(rel)
            buffered_pages = self._cache_state.get(rel, -1)
            if buffered_pages == -1:
                buffered_pages = self._stats.n_buffered(rel)
                self._cache_state[rel] = buffered_pages
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
    """The featurizer is used to transform query plans into their corresponding feature vectors.

    In addition to the plan featurization, we also use the featurizer to store normalization data,
    specifically regarding the predicted plan runtimes.

    Bao uses a featurization scheme that is independent of a particular database instance/schema.
    However, the featurization can still benefit from prior knowledge about the target workload,
    because this allows to generate smaller feature vectors and to use better min-max normalization
    ranges.
    Accordingly, we provide two different methods to create new featurization instances:

    - `online` creates a featurizer exclusively based on the database, i.e. without any leaks from
      the target workload. While this is arguably the most realistic setting, the resulting feature
      vectors will be the most difficult to learn.
    - `infer_from_samples` creates a featurizer tailored to a specific test workload.

    The original Bao paper did not specify, how their featurization is created. So choose whichever
    strategy works best for you.

    Once a featurization has been created, use `encode_plan` to transform query plans into their
    corresponding feature vectors. The other ``encode_XYZ`` methods handle the different sub-steps
    of the featurization process. Usually, there is no need to call these explicitly.

    An existing featurization can be persisted using the `store` method. As a convenience function,
    the `load_or_build` function will load a previously stored featurization. If it does not exist,
    it will be created and stored.
    """

    @staticmethod
    def online(
        database: pb.Database, *, max_runtime_ms: float = 1000 * 60 * 60
    ) -> BaoFeaturizer:
        """Infers the featurization from the database.

        This featurization strategy trades off high generality for potentially larger
        (and thus more sparse) feature vectors.

        Notes
        -----
        We use the following rules to determine the featurization parameters: All node types that
        appear as valid Postgres operators are included in the operator encoding. The maximum
        cardinality and cost are derived by planning an SQL query and obtaining the corresponding
        estimates from the target database. The query performs a cross product between the three
        largest tables in the schema.
        The maximum runtime can be specified by the user. As a default, we use one hour.
        """
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
            database=database,
        )

    @staticmethod
    def infer_from_sample(
        sample: pd.DataFrame, *, database: pb.Database, plan_col: str = "query_plan"
    ) -> BaoFeaturizer:
        """Builds a featurizer tailored to a specific set of training samples.

        This featurization strategy trades off generality for smaller (and thus less sparse) feature
        vectors.

        Parameters
        ----------
        sample: pd.DataFrame
            The training samples. The query plans should be proper `pb.QueryPlan` instances. If they
            are not, they are parsed under the assumption that they were obtained from the same
            database system as the target database.
        database: pb.Database
            The target database
        plan_col: str
            The column in the `sample` DataFrame that contains the query plans

        Notes
        -----
        We use the following rules to obtain the featurization parameters: The operator encoding
        includes all operators that appear in the sample plans. The maximum cardinality, cost, and
        runtime all also retrieved directly from the query plans. Therefore, all contained plans
        should be valid EXPLAIN ANALYZE plans.
        """
        if not isinstance(sample[plan_col].iloc[0], pb.QueryPlan):
            plans = sample[plan_col].map(database.optimizer().parse_plan)
        else:
            plans = sample[plan_col]

        allowed_ops: set[str] = set()
        min_card, max_card = np.inf, 0
        min_cost, max_cost = np.inf, 0
        min_runtime, max_runtime = np.inf, 0

        for plan in plans:
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
            database=database,
        )

    @staticmethod
    def pre_built(archive: Path | str, *, database: pb.Database) -> BaoFeaturizer:
        """Loads a previously built featurization from disk.

        See Also
        --------
        store : inverse method to persist a featurization to disk
        """
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
            database=database,
        )

    @staticmethod
    def load_or_build(archive: Path | str, *, database: pb.Database) -> BaoFeaturizer:
        """Integrated inference and storage procedure.

        If the featurization has already been stored to `archive`, it will simply be loaded.
        Otherwise, it will be inferred using `online`.
        """
        archive = Path(archive)
        if archive.is_file():
            return BaoFeaturizer.pre_built(archive, database=database)

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
        database: pb.Database,
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

        self._db = database

    @property
    def database(self) -> pb.Database:
        """Get the current target database"""
        return self._db

    @property
    def out_shape(self) -> int:
        """Get the number of elements each node-level vector will have."""
        return (
            len(self._op_enc.categories_[0])
            + 1  # card enc
            + 1  # cost enc
            + 1  # cache_pct enc
        )

    def encode(self, node: BinarizedQep) -> np.ndarray:
        """Transforms a specific pre-processed node into its feature representation."""
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
        cache_state: DatabaseCacheState | None = None,
    ) -> FeaturizedNode | tuple:
        """Transforms a raw PostBOUND query plan into its feature representation.

        If the cache state is not explicitly given, it will be inferred from the target database.
        """
        if cache_state is None:
            cache_state = DatabaseCacheState(self._db)

        binarized = BinarizedQep.create_for(plan, cache_state=cache_state)
        return self._featurize_qep(binarized)

    def encode_operator(self, node: BinarizedQep) -> np.ndarray:
        """Applies the operator one-hot encoding to the node."""
        operator = "__bao_null__" if node.is_dummy else node.node
        operator = _PGNodeMap.get(operator, operator)
        enc = self._op_enc.transform([(operator,)])
        return enc[0]

    def encode_cardinality(self, node: BinarizedQep) -> np.ndarray:
        """Min-max scales the cardinality estimate of the node."""
        if node.is_dummy:
            return np.zeros(1)
        scaled = np.log(node.cardinality + 1) - self._min_card
        scaled /= self._max_card - self._min_card
        return np.asarray([scaled])

    def encode_cost(self, node: BinarizedQep) -> np.ndarray:
        """Min-max scales the cost estimate of the node."""
        if node.is_dummy:
            return np.zeros(1)
        scaled = np.log(node.cost + 1) - self._min_cost
        scaled /= self._max_cost - self._min_cost
        return np.asarray([scaled])

    def encode_cache_pct(self, node: BinarizedQep) -> np.ndarray:
        """Transforms the cache percentage of the node.

        Since the cache percentage is already a number in [0, 1], we don't need to do anything for
        normal nodes. Dummies still have to be processed accordingly.
        Nevertheless, this function primarily exists to have a uniform encoding pattern for the
        different attributes of a node, not because we apply any fancy featurization logic.
        """
        if node.is_dummy:
            return np.zeros(1)
        return np.asarray([node.cache_pct])

    def transform_runtime(self, runtime) -> np.ndarray:
        """Scales a raw runtime measurement."""
        return self._runtime_pipeline.transform(runtime)

    def inverse_transform_runtime(self, scaled):
        """Undos the runtime scaling."""
        return self._runtime_pipeline.inverse_transform(scaled)

    def store(self, archive: Path | str) -> None:
        """Persists the featurization info at the specified location.

        The `archive` is assumed to be JSON file.
        """
        archive = Path(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)

        serialized = {
            "allowed_ops": self._allowed_ops,
            "min_card": self._min_card,
            "max_card": self._max_card,
            "min_cost": self._min_cost,
            "max_cost": self._max_cost,
            "max_runtime_ms": self._max_runtime,
            "min_runtime_ms": self._min_runtime,
        }
        with open(archive, "w") as f:
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
