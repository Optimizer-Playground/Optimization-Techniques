"""
Bao PostBOUND adapter
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

import functools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import postbound as pb
import torch
from tqdm import tqdm

from ..util import wrap_logger
from ._experience import BaoExperience
from ._featurizer import BaoFeaturizer, DatabaseCacheState
from ._model import BaoModel


class HintSetSpec(TypedDict):
    hash: bool
    merge: bool
    nlj: bool
    seq: bool
    idx: bool
    idx_o: bool


def _as_hint_set(options: HintSetSpec) -> pb.PhysicalOperatorAssignment:
    hint_set = pb.PhysicalOperatorAssignment()

    hint_set.set(pb.JoinOperator.HashJoin, options.get("hash", True))
    hint_set.set(pb.JoinOperator.SortMergeJoin, options.get("merge", True))
    hint_set.set(pb.JoinOperator.NestedLoopJoin, options.get("nlj", True))
    hint_set.set(pb.ScanOperator.SequentialScan, options.get("seq", True))
    hint_set.set(pb.ScanOperator.IndexScan, options.get("idx", True))
    hint_set.set(pb.ScanOperator.IndexOnlyScan, options.get("idx_o", True))

    return hint_set


def _stringify_hint_set(hint_set: pb.PhysicalOperatorAssignment) -> str:
    disabled_components: list[pb.PhysicalOperator] = []
    for operator, enabled in hint_set.global_settings.items():
        if enabled:
            continue
        disabled_components.append(operator)

    if not disabled_components:
        return "<default>"

    disabled_txt = ", ".join(hint.name for hint in disabled_components)
    return f"no {disabled_txt}"


def default_hint_sets() -> list[pb.PhysicalOperatorAssignment]:
    """The hint sets used by the reference implementation of Bao.

    These hint sets are guaranteed to work with Postgres, support for other database systems
    might vary.
    """
    arms: list[pb.PhysicalOperatorAssignment] = []

    # case 0
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": True,
                "seq": True,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 1
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": False,
                "seq": True,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 2
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": True,
                "seq": True,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 3
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": False,
                "seq": True,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 4
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": True,
                "seq": True,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 5
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": True,
                "seq": False,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 6
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 7
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": True,
                "nlj": True,
                "seq": False,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 8
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": False,
                "seq": False,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 9
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 10
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": False,
                "seq": True,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 11
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": True,
                "seq": True,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 12
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": False,
                "seq": True,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 13
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 14
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": False,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 15
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": True,
                "nlj": True,
                "seq": True,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 16
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": False,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 17
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 18
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": True,
                "nlj": True,
                "seq": False,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 19
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": True,
                "nlj": True,
                "seq": True,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 20
    arms.append(
        _as_hint_set(
            {
                "hash": False,
                "merge": False,
                "nlj": True,
                "seq": True,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 21
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": False,
                "seq": False,
                "idx": True,
                "idx_o": True,
            }
        )
    )

    # case 22
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": True,
                "nlj": False,
                "seq": False,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    # case 23
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": True,
                "seq": True,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 24
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": False,
                "seq": False,
                "idx": True,
                "idx_o": False,
            }
        )
    )

    # case 25
    arms.append(
        _as_hint_set(
            {
                "hash": True,
                "merge": False,
                "nlj": True,
                "seq": False,
                "idx": False,
                "idx_o": True,
            }
        )
    )

    return arms


class BaoOptimizer(
    pb.CompleteOptimizationAlgorithm, pb.PhysicalOperatorSelection, pb.CostModel
):
    """Bao is a learned query optimizer that uses hint sets to select optimal candidate plans.

    Bao sits on top of a native database system and steers its native optimizer with hint sets.
    Each hint set restricts the optimizer's search space, which results in a different execution
    plan being selected. Afterwards, Bao uses a learned model to assess the quality of the different
    candidate plans and ultimately selects the best one.

    The prediction process is handled by the `BaoModel` class. It receives query plans that have
    been prepared by a `BaoFeaturizer`. To ensure that the cost predictions are accurate, Bao uses
    reinforcement learning to feed execution statistics of past query plans (stored as a
    `BaoExperience`) back into the model.
    This behavior is controlled by the different retraining parameters.

    Creating a new optimizer
    ------------------------
    A fresh optimizer can be obtained by simply creating a new instance of this class. Due to the
    reinforcement learning architecture, it is important to pass execution traces of optimized
    queries back to the model. This process can be bootstrapped with the `calibrate` method which
    trains the model in an offline fashion. Suitable training queries can be generated using the
    *query-sampler.py* tool.

    The `load_or_build` method provides a convenient way to either load an existing model from disk
    (if it exists), or to initialize a new one. This new model will then be stored at the designated
    location.

    Parameters
    ----------
    target_db: pb.Database
        The database system that Bao should wrap. This system needs to provide basic hinting and
        optimizer support to obtain different candidate plans.
    hint_sets: Optional[Iterable[pb.PhysicalOperatorAssignment]]
        The hint sets that Bao should use to obtain different candidate plans. It is the user's
        responsibility to make sure that the `target_db` actually supports these hint sets.
        By default, we use the same 26 hint sets that the reference implementation uses.
    tcnn: Optional[BaoModel]
        The cost model that should be used to assess the candidate plans. If this parameter is
        omitted, a new model with random weights will be used.
    featurizer: Optional[BaoFeaturizer]
        The featurizer that transform query plans into feature vectors. By default, this will be
        inferred based on the target database (see `BaoFeaturizer.online`). If a featurizer and a
        TCNN are explicitly provided, it is the user's responsibility to ensure that they are
        compatible, i.e. that the featurizer provides vectors of the correct shape for the TCNN.
    retrain: bool
        Whether the reinforcement learning loop should be active. If this is enabled, the TCNN model
        will be periodically retrained once enough experience (i.e. past query executions) have been
        collected. If this is disabled, the model is frozen. Notice that disabling retraining still
        allows for new experience to be added. This experience will simply not be used until
        retraining is enabled again.
    training_epochs: int
        The number of epochs that should be used for each retraining phase. By default, we use
        100 epochs, which is the same as the reference implementation.
    experience_window: int
        The maximum number of past query executions that should be considered for retraining. If
        more queries arrive, the oldest ones will be dropped. Even if retraining is currently
        disabled, the experience buffer will still be updated with new query executions.
        By default, we use the last 2000 queries, which is the same as the reference implementation.
    retraining_frequency: int
        How often retraining should be triggered. By default, the TCNN will be retrained after every
        100 new queries, which is the same as the reference implementation. If retraining is
        disabled, this parameter has no effect.
    verbose: bool | pb.util.Logger
        Whether to print verbose output during optimization and training. If a `pb.util.Logger` is
        provided, it will be used as-is. Otherwise, a default logger will be created depending on
        the bool.

    Attributes
    ----------
    retrain: bool
        Whether the reinforcement learning loop is active. This can be toggled on and off at runtime.
        Notice that disabling retraining has no effect on the experience buffer. It will be
        maintained as usual, just without triggering the retraining process.

    See Also
    --------
    BaoModel: The TCNN model used for cost prediction.
    BaoFeaturizer: The featurizer used to prepare query plans for the TCNN
    BaoExperience: The experience buffer that stores past query executions and triggers retraining.

    References
    ----------
    .. Ryan Marcus et al.: "Bao: Making Learned Query Optimization Practical"
       (SIGMOD 2021), https://doi.org/10.1145/3448016.3452838
    """

    @staticmethod
    def pre_trained(
        archive: Path | str,
        *,
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> BaoOptimizer:
        """Loads a Bao model from disk.

        This is the inverse operation to `store`. Since `store` also takes care of persisting the
        featurizer and experience, it is loaded as part of this process.
        The `archive` must either point to a valid JSON file containing the model info, or the
        directory that contains the catalog.

        See Also
        ---------
        store : The inverse operation to this method. Persists a Bao model to disk.
        """
        archive = Path(archive)
        if archive.is_dir():
            archive = archive / "catalog.json"

        with open(archive, "r") as f:
            catalog = json.load(f)

        featurizer = BaoFeaturizer.pre_built(catalog["featurizer"], database=database)
        experience = BaoExperience.load(
            catalog["experience"]["catalog"], featurizer=featurizer
        )

        weights = torch.load(catalog["tcnn_model"])
        model = BaoModel(featurizer.out_shape)
        model.load_state_dict(weights)

        hint_sets = [
            pb.opt.read_operator_assignment_json(hints)
            for hints in catalog["hint_sets"]
        ]

        bao = BaoOptimizer(
            database,
            hint_sets=hint_sets,
            tcnn=model,
            featurizer=featurizer,
            experience=experience,
            training_epochs=catalog["training"]["epochs"],
            retrain=False,
            verbose=verbose,
        )
        return bao

    @staticmethod
    def load_or_build(
        archive: Path | str,
        *,
        database: pb.Database,
        calibration_queries: pb.Workload | pd.DataFrame,
        retrain: bool = False,
        training_epochs: int = 100,
        experience_window: int = 2000,
        retraining_frequency: int = 100,
        verbose: bool | pb.util.Logger = False,
    ) -> BaoOptimizer:
        """Integrated model training and storage procedure.

        If a Bao model has already been stored to `archive`, it will simply be loaded.
        Otherwise, a new model will be created and trained using the `calibration_queries`.
        The featurizer is inferred based on the database.

        The training process can be customized using the same parameter as the `BaoOptimizer`
        constructor. Notice that the `retrain` parameter only controls whether the reinforcement
        learning loop will be active after the initial training phase. The calibration will always
        happen.

        Once the model has been trained, it will be stored at `archive` along with the
        featurizer and experience.

        See Also
        ---------
        BaoFeaturizer.online : Inference logic for the featurizer
        """
        log = wrap_logger(verbose)
        archive = Path(archive)
        if archive.is_dir():
            archive = archive / "catalog.json"

        if archive.is_file():
            log(f"Detected existing BAO model at {archive}. Re-loading.")
            return BaoOptimizer.pre_trained(archive, database=database, verbose=verbose)

        log(f"No BAO model found at {archive}. Creating a new one.")
        bao = BaoOptimizer(
            database,
            training_epochs=training_epochs,
            retrain=retrain,
            experience_window=experience_window,
            retraining_frequency=retraining_frequency,
            verbose=verbose,
        )

        log("Calibrating new BAO model")
        bao.calibrate(calibration_queries)

        log("Storing model")
        bao.store(archive)
        return bao

    def __init__(
        self,
        target_db: pb.Database,
        *,
        hint_sets: Optional[Iterable[pb.PhysicalOperatorAssignment]] = None,
        tcnn: Optional[BaoModel] = None,
        featurizer: Optional[BaoFeaturizer] = None,
        experience: Optional[BaoExperience] = None,
        retrain: bool = True,
        training_epochs: int = 100,
        experience_window: int = 2000,
        retraining_frequency: int = 100,
        verbose: bool | pb.util.Logger = False,
    ) -> None:
        super().__init__()
        self._db = target_db
        self._hint_sets = (
            list(hint_sets) if hint_sets is not None else default_hint_sets()
        )
        self._featurizer = featurizer or BaoFeaturizer.online(self._db)
        self._experience = experience or BaoExperience(
            self._featurizer,
            sample_window=experience_window,
            retraining_frequency=retraining_frequency,
        )
        self._tcnn = tcnn or BaoModel(self._featurizer.out_shape)
        self._retrain = retrain
        self._epochs = training_epochs
        self._log = wrap_logger(verbose)
        self._verbose = bool(verbose)

    @property
    def retrain(self) -> bool:
        """Get/update the current retraining status.

        Note that having retraining off does not disabled gathering new experience.
        Samples are still recorded and the experience storage is still updated, only the retraining
        process is ignored.
        """
        return self._retrain

    @retrain.setter
    def retrain(self, retrain: bool) -> None:
        self._retrain = retrain

    def optimize_query(self, query: pb.SqlQuery) -> pb.QueryPlan:
        cache_state = DatabaseCacheState(self._db)
        plans = [self._generate_plan(query, hint_set) for hint_set in self._hint_sets]
        featurized = [
            self._featurizer.encode_plan(plan, cache_state=cache_state)
            for plan in plans
        ]

        predictions = self._tcnn(featurized)
        idxmin = int(torch.argmin(predictions).item())

        hint_set = _stringify_hint_set(self._hint_sets[idxmin])
        self._log(f"Selected arm {idxmin} ({hint_set}) for query {query}")
        return plans[idxmin]

    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: Optional[pb.JoinTree]
    ) -> pb.PhysicalOperatorAssignment:
        cache_state = DatabaseCacheState(self._db)
        plans = [
            self._generate_plan(query, hint_set, join_order=join_order)
            for hint_set in self._hint_sets
        ]

        featurized = [
            self._featurizer.encode_plan(plan, cache_state=cache_state)
            for plan in plans
        ]
        predictions = self._tcnn(featurized)

        idxmin = int(torch.argmin(predictions).item())
        hint_set = self._hint_sets[idxmin]
        self._log(
            f"Selected arm {idxmin} ({_stringify_hint_set(hint_set)}) for query {query}"
        )
        return hint_set

    def estimate_cost(self, query: pb.SqlQuery, plan: pb.QueryPlan) -> pb.Cost:
        cache_state = DatabaseCacheState(self._db)
        featurized = self._featurizer.encode_plan(plan, cache_state=cache_state)
        prediction = self._tcnn([featurized])
        return prediction.item()

    def add_experience(
        self, plan: pb.QueryPlan, runtime_ms: float | None = None
    ) -> None:
        """Records a new experience sample, possibly triggering retraining.

        If the retraining threshold is reached, a new TCNN model is retrained (provided that it is
        not disabled). In any case, the new sample is added to the experience storage. If the
        storage is full, the oldest sample will be evicted.

        The runtime can be omitted, in which case it will be inferred from the plan's execution
        time.
        """
        if runtime_ms is None:
            runtime_ms = plan.execution_time * 1000

        self._experience.add(plan, runtime_ms)
        if not self._retrain or not self._experience.should_retrain():
            return

        self._log("Updating model")
        self._log("Preparing experience")
        dataset = self._experience.samples()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, collate_fn=lambda xs: xs
        )  # default collation messes up the data types due to namedtuple - use our own dummy

        self._log("Obtaining new model")
        self._tcnn = BaoModel(self._featurizer.out_shape)
        self._train(loader)
        self._experience.mark_retrained()

    def calibrate(
        self,
        workload: pb.Workload | pd.DataFrame,
        *,
        plan_column: str = "query_plan",
        runtime_column: str = "runtime_ms",
        timeout_ms: Optional[float] = None,
        from_scratch: bool = False,
    ) -> None:
        """Performs a batch training of the TCNN.

        This function operates in two different modes: If the workload is given as a set of queries,
        these will be executed and their query plans and runtimes are gathered. In this mode, an
        optional timeout can be supplied to skip very long running queries.
        If the workload is instead given as a Pandas DataFrame, it is assumed that this workload
        contains query plans and runtimes that have already been calculated in an offline training
        phase (e.g., using the *query-sampler.py* tool).

        Once all query plans have been gathered, the TCNN is trained. By default, the current model
        weights are used as a starting point. However, this can be disabled by setting
        `from_scratch` to True. In this case, a new TCNN is initialized.

        Note that this method modifies the contents of the experience store.

        Parameters
        ----------
        workload: pb.Workload | pd.DataFrame
            The workload that should be used for calibration. This can either be a set of queries
            to execute, or a DataFrame containing query plans and runtimes.
        plan_column: str
            If `workload` is a DataFrame, this column is expected to contain the query
            plans. If the column does not already contain `pb.QueryPlan` instances, they will be
            parsed under the assumption that the plans have been generated by the same database
            system as the target database. Defaults to "query_plan".
        runtime_column: str
            If `workload` is a DataFrame, this column is expected to contain the runtimes (in
            milliseconds) of the query plans. Defaults to "runtime_ms".
        timeout_ms: Optional[float]
            If `workload` is a set of queries, this timeout (in milliseconds) will be applied to
            each query execution. If a query exceeds this runtime, it will be skipped and not
            added to the experience.
        from_scratch: bool
            Whether to start the training with a fresh TCNN model. Defaults to False, which
            re-uses the current model weights as a starting point.
        """
        if from_scratch:
            self._log("Obtaining new model")
            self._tcnn = BaoModel(self._featurizer.out_shape)

        if isinstance(workload, pd.DataFrame):
            self._retrain_offline(
                workload, plan_col=plan_column, runtime_col=runtime_column
            )
            return

        self._log("Gathering query plans for calibration queries")
        query_iter = (
            tqdm(workload.queries(), desc="Training query", unit="q")
            if self._verbose
            else workload.queries()
        )

        if timeout_ms and not isinstance(self._db, pb.db.TimeoutSupport):
            raise ValueError("Target database system does not provide timeout support.")

        if timeout_ms:
            assert isinstance(self._db, pb.db.TimeoutSupport)
            timeout = 1000 * timeout_ms if timeout_ms else None
            executor = functools.partial(self._db.execute_with_timeout, timeout=timeout)
        else:
            executor = functools.partial(self._db.execute_query, raw=True)

        for query in query_iter:
            query = pb.transform.as_explain_analyze(query)
            result_set = executor(query)
            if result_set is None:
                continue

            raw_plan = result_set
            plan = self._db.optimizer().parse_plan(raw_plan, query=query)
            runtime = plan.execution_time
            self._experience.add(plan, runtime * 1000)

        dataset = self._experience.samples()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, collate_fn=lambda xs: xs
        )  # default collation messes up the data types due to namedtuple - use our own dummy
        self._train(loader)
        self._experience.mark_retrained()

    def bench_feedback(self, result: pb.bench.ExecutionResult) -> None:
        """Callback method suitable for `execute_workload` to feed the reinforcement-learning loop.

        If retraining is enabled, this can trigger a new training pass once the retraining
        threshold is reached.

        Examples
        --------
        >>> bao = ...
        >>> pb.bench.execute_workload(workload, on=bao, exec_callback=bao.bench_feedback)
        """
        if result.status != "ok":
            return
        query_plan = self._db.optimizer().parse_plan(result.query_result)
        self.add_experience(query_plan, 1000 * result.execution_time)

    def store(self, archive_dir: Path | str) -> None:
        """Persists an optimizer and all related data to disk.

        This will persist the selected hint sets, the featurizer, the TCNN weights and the samples
        in the experience storage.

        Parameters
        ----------
        archive_dir: Path | str
            The directory where the optimizer and all related data will be stored. This directory
            should be exclusive to the current optimizer. If two optimizers share the same
            directory, they will overwrite each others data.
        """
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        featurizer_catalog = archive_dir / "featurizer.json"
        self._log("Exporting featurizer to", featurizer_catalog)
        self._featurizer.store(featurizer_catalog)

        experience_catalog = archive_dir / "experience-catalog.json"
        experience_store = archive_dir / "experience-store.parquet"
        self._log("Exporting experience to", experience_catalog)
        self._experience.store(
            experience_catalog,
            experience_path=experience_store,
        )

        model_path = archive_dir / "tcnn.pt"
        self._log("Storing TCNN model to", model_path)
        weights = self._tcnn.state_dict()
        torch.save(weights, model_path)

        serialized = {
            "hint_sets": self._hint_sets,
            "training": {"epochs": self._epochs},
            "featurizer": featurizer_catalog,
            "experience": {
                "catalog": experience_catalog,
                "store": experience_store,
            },
            "tcnn_model": model_path,
        }
        self._log("Creating catalog")
        with open(archive_dir / "catalog.json", "w") as f:
            pb.util.to_json_dump(serialized, f)

    def describe(self) -> pb.util.jsondict:
        return {
            "name": "BAO",
            "database": self._db.describe(),
            "hint_sets": self._hint_sets,
        }

    def _generate_plan(
        self,
        query: pb.SqlQuery,
        hint_set: pb.PhysicalOperatorAssignment,
        *,
        join_order: Optional[pb.JoinTree] = None,
    ) -> pb.QueryPlan:
        hinted_query = self._db.hinting().generate_hints(
            query, join_order=join_order, physical_operators=hint_set
        )
        return self._db.optimizer().query_plan(hinted_query)

    def _retrain_offline(
        self, samples: pd.DataFrame, *, plan_col: str, runtime_col: str
    ) -> None:
        """Utility to prepare and invoke training based on offline samples."""
        if isinstance(samples[plan_col].iloc[0], pb.QueryPlan):
            plans = samples[plan_col]
        else:
            plans = samples[plan_col].map(self._db.optimizer().parse_plan)
        runtimes = samples[runtime_col]

        for i in range(len(samples)):
            self._experience.add(plans[i], runtimes[i])

        dataset = self._experience.samples()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, collate_fn=lambda xs: xs
        )  # default collation messes up the data types due to namedtuple - use our own dummy
        self._train(loader)
        self._experience.mark_retrained()

    def _train(self, samples: torch.utils.data.DataLoader) -> None:
        self._tcnn.train()
        optimizer = torch.optim.Adam(self._tcnn.parameters())
        mse_loss = torch.nn.MSELoss()

        self._log("Starting training")
        for epoch in range(self._epochs):
            loss_total = 0.0

            for batch in samples:
                xs, y = list(zip(*batch))
                y = torch.Tensor(np.array(y)).reshape(-1, 1)
                optimizer.zero_grad()
                prediction = self._tcnn.forward(xs)
                loss = mse_loss(prediction, y)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()

            epoch_str = str(epoch + 1).rjust(len(str(self._epochs)))
            self._log(f"Epoch: {epoch_str} / {self._epochs} :: loss = {loss_total}")

        self._tcnn.eval()
