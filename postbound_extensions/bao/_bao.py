from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import postbound as pb
import torch
from tqdm import tqdm

from ..util import wrap_logger
from ._experience import BaoExperience
from ._featurizer import BaoFeaturizer
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


class BaoOptimizer(pb.CompleteOptimizationAlgorithm):
    @staticmethod
    def pre_trained(
        archive: Path | str,
        *,
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> BaoOptimizer:
        archive = Path(archive)
        if archive.is_dir():
            archive = archive / "catalog.json"

        with open(archive, "r") as f:
            catalog = json.load(f)

        featurizer = BaoFeaturizer.pre_built(catalog["featurizer"])
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
        calibration_queries: pb.Workload,
        retrain: bool = False,
        training_epochs: int = 100,
        experience_window: int = 2000,
        retraining_frequency: int = 100,
        verbose: bool | pb.util.Logger = False,
    ) -> BaoOptimizer:
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
        tcnn: Optional[BaoModel | torch.fx.GraphModule] = None,
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
        return self._retrain

    @retrain.setter
    def retrain(self, retrain: bool) -> None:
        self._retrain = retrain

    def optimize_query(self, query: pb.SqlQuery) -> pb.QueryPlan:
        plans: list[pb.QueryPlan] = [
            self._generate_plan(query, hint_set) for hint_set in self._hint_sets
        ]
        featurized = [
            self._featurizer.encode_plan(plan, database=self._db) for plan in plans
        ]

        predictions = self._tcnn(featurized)
        idxmin = int(torch.argmin(predictions).item())

        hint_set = _stringify_hint_set(self._hint_sets[idxmin])
        self._log(f"Selected arm {idxmin} ({hint_set}) for query {query}")
        return plans[idxmin]

    def add_experience(self, plan: pb.QueryPlan, runtime_ms: float) -> None:
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

    def calibrate(
        self, workload: pb.Workload, *, timeout_ms: Optional[float] = None
    ) -> None:
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

    def store(self, archive_dir: Path | str) -> None:
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
        self, query: pb.SqlQuery, hint_set: pb.PhysicalOperatorAssignment
    ) -> pb.QueryPlan:
        hinted_query = self._db.hinting().generate_hints(
            query, physical_operators=hint_set
        )
        return self._db.optimizer().query_plan(hinted_query)

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
