"""
Bao Experience Store
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pandas as pd
import postbound as pb
import torch
from torch.utils.data import Dataset

from ._cache import DatabaseCacheState
from ._featurizer import BaoFeaturizer


def read_cache_state(serialized: DatabaseCacheState | list | str) -> DatabaseCacheState:
    """Parses a cache state, either given as a mapping or in the serialized form of `BaoSample.serialize()`."""
    if isinstance(serialized, str):
        serialized = json.loads(serialized)
    if isinstance(serialized, Mapping):
        return {
            (pb.parser.load_table_json(rel) if not isinstance(rel, pb.TableReference) else rel): pct
            for rel, pct in serialized.items()
        }
    return {pb.parser.load_table_json(entry["relation"]): entry["buffer_pct"] for entry in serialized}


@dataclass
class BaoSample:
    """A single entry of the experience storage.

    Attributes / Parameters
    -------------------------
    plan : pb.QueryPlan
        The query plan that was executed
    runtime_ms : float
        The observed plan runtime in milliseconds
    cache_state : DatabaseCacheState
        The state of the database cache at the time of query execution.
    """

    plan: pb.QueryPlan
    cache_state: DatabaseCacheState
    runtime_ms: float

    # (featurizer, (features, scaled runtime)) - the encoding only depends on the sample and the featurizer, so we
    # compute it once instead of in every training epoch. It is never persisted.
    _encoded: tuple[BaoFeaturizer, tuple] | None = field(default=None, init=False, repr=False, compare=False)

    def serialize(self) -> dict:
        serialized_cache_state = [{"relation": rel, "buffer_pct": pct} for rel, pct in self.cache_state.items()]
        return {
            "plan": pb.util.to_json(self.plan),
            "cache_state": serialized_cache_state,
            "runtime_ms": self.runtime_ms,
        }


class BaoExperience(Dataset[tuple[torch.Tensor, torch.Tensor, float]]):
    """BaoExperience stores past query plans and their observed runtimes.

    An experience store has a limited capacity. If new samples are added beyond this capacity,
    the oldest samples are automatically discarded.
    In addition, the experience keeps track of the required number of new samples before the TCNN
    should be updated.

    The current experience can be manually stored and loaded similar to the actual optimizer.
    However, the optimizer itself also manages the experience. Therefore, there is little reason to
    manually invoke this functionality on the experience.

    An experience store can be directly used as a PyTorch training dataset. To suport this
    functionality, the experience needs access to the featurizer.

    Parameters
    -----------
    featurizer : BaoFeaturizer
        The featurizer is required to transform the query plans into feature vectors.
    sample_window : int, optional
        The maximum number of samples that are stored in the experience. If the number of samples
        exceeds this limit, the oldest samples are discarded. Default: 2000, which is the window
        size recommended in the original Bao paper.
    retraining_frequency : int, optional
        The number of new samples that need to be added to the experience before the TCNN should
        be retrained. Default: 100, which is the retraining frequency recommended in the original
        Bao paper.
    existing_samples : Iterable[BaoSample], optional
        Samples that should be added to the experience upon creation.
    log : pb.util.Logger, optional
        Logger for diagnostic messages during featurization.

    Notes
    -----
    Each sample stores the cache state of the database at the time the plan was selected. The optimizer registers
    this state via `register_cache_state` whenever it produces a plan (`calibrate`, `optimize_query`,
    `select_physical_operators`). Adding a plan whose cache state was never registered is an error: the cache state
    is never inferred after the fact.
    """

    @staticmethod
    def load(archive: Path | str, *, featurizer: BaoFeaturizer, log: pb.util.Logger | None = None) -> BaoExperience:
        """Loads previously stored experience."""
        with open(archive, "r") as f:
            catalog = json.load(f)

        sample_df = pb.util.read_df(catalog["experience_archive"])
        samples: list[BaoSample] = []
        for row in sample_df.itertuples():
            plan = pb.opt.read_query_plan_json(row.plan)  # type: ignore
            cache_state = read_cache_state(row.cache_state)  # type: ignore
            runtime = row.runtime_ms  # type: ignore
            samples.append(BaoSample(plan, cache_state, runtime))

        return BaoExperience(
            featurizer,
            sample_window=catalog["window_size"],
            retraining_frequency=catalog["retrain_frequency"],
            existing_samples=samples,
            log=log,
        )

    @staticmethod
    def load_or_build(archive: Path | str, *, featurizer: BaoFeaturizer) -> BaoExperience:
        """Loads experience from archive if it exists. Otherwise, creates a new experience."""
        archive = Path(archive)
        return BaoExperience.load(archive, featurizer=featurizer) if archive.is_file() else BaoExperience(featurizer)

    def __init__(
        self,
        featurizer: BaoFeaturizer,
        *,
        sample_window: int = 2000,
        retraining_frequency: int = 100,
        existing_samples: Iterable[BaoSample] | None = None,
        log: pb.util.Logger | None = None,
    ) -> None:
        self._log = log or pb.util.standard_logger(False)

        self._featurizer = featurizer
        self._window = sample_window
        self._retrain_freq = retraining_frequency

        self._storage: collections.deque[BaoSample] = collections.deque(maxlen=self._window)
        if existing_samples is not None:
            self._storage.extend(existing_samples)

        self._new_samples = 0
        # Cache states of plans that were produced by the optimizer but whose runtime is not yet known. We do not remove
        # entries once the plan's runtime is added, because the same plan might be executed multiple times (e.g. by
        # repeated benchmark runs). Instead, the oldest entries are evicted once the capacity is exceeded.
        self._cache_state: collections.OrderedDict[pb.QueryPlan, DatabaseCacheState] = collections.OrderedDict()

    def add(self, plan: pb.QueryPlan | BaoSample, runtime_ms: float | None = None) -> None:
        """Store a new sample in the experience.

        If the sample is given as a `BaoSample`, the runtime can be omitted. For plain query plans,
        it is required.
        """
        if isinstance(plan, BaoSample):
            self._storage.append(plan)
            self._new_samples += 1
            return

        if runtime_ms is None:
            raise ValueError("runtime_ms is required if plan is supplied as a QueryPlan")
        elif not isinstance(plan, pb.QueryPlan):
            raise ValueError("plan must be a QueryPlan")

        cache_state = self._cache_state.get(plan)
        if cache_state is None:
            raise ValueError(
                f"No cache state registered for plan {plan}. Cannot add to experience. "
                "BaoOptimizer should take care of this automatically. Make sure to only add samples via its `calibrate` method "
                "or the reinforcement learning loop "
                "(getting a plan from the optimizer and then adding its runtime back to the experience)."
            )
        sample = BaoSample(plan, cache_state, runtime_ms)
        self._storage.append(sample)
        self._new_samples += 1

    def register_cache_state(self, plan: pb.QueryPlan, cache_state: DatabaseCacheState) -> None:
        """Registers the cache state for a given query plan.

        Plans are matched by their structure (node types, base tables and children), i.e. plans that are structurally
        identical share the same cache state. The most recent registration wins.
        """
        self._cache_state[plan] = cache_state
        self._cache_state.move_to_end(plan)
        while len(self._cache_state) > self._window:
            self._cache_state.popitem(last=False)

    def should_retrain(self) -> bool:
        """Whether enough new samples have been added to justify retraining the TCNN."""
        return self._new_samples >= self._retrain_freq

    def samples(self) -> Dataset[tuple[torch.Tensor, torch.Tensor, float]]:
        """Provides the current experience as a PyTorch dataset of (features, runtimes) tuples."""
        return self

    def mark_retrained(self) -> None:
        """Marks the experience as having been used for retraining.

        This resets the internal new sample counter. This is called by the optimizer after
        retraining is completed. There is generally no need to call this method manually.
        """
        self._new_samples = 0

    def clear(self) -> None:
        """Removes all samples from the experience."""
        self._storage.clear()
        self._cache_state.clear()
        self._new_samples = 0

    def store(self, catalog_path: Path | str, *, experience_path: Path | str | None = None) -> None:
        """Persists the current experience to disk.

        Parameters
        -----------
        catalog_path : Path | str
            The path where the experience metadata/index should be stored.
        experience_path : Optional[Path | str], optional
            The path where the actual experience samples should be stored. If not given, the samples
            will be stored in the same directory as the catalog with the name "experience.parquet".
        """
        catalog_path = Path(catalog_path)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        experience_path = experience_path or (catalog_path.parent / "experience.parquet")
        experience_path = Path(experience_path)
        experience_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([sample.serialize() for sample in self._storage])
        pb.util.write_df(df, experience_path)

        serialized = {
            "window_size": self._window,
            "retrain_frequency": self._retrain_freq,
            "experience_archive": str(experience_path),
        }
        with open(catalog_path, "w") as f:
            json.dump(serialized, f)

    def __len__(self) -> int:
        return len(self._storage)

    @override
    def __getitem__(self, index):
        sample = self._storage[index]
        if sample._encoded is not None and sample._encoded[0] is self._featurizer:
            return sample._encoded[1]

        featurized = self._featurizer.encode_plan(sample.plan, cache_state=sample.cache_state, log=self._log)
        scaled_runtime = self._featurizer.transform_runtime([(sample.runtime_ms,)])
        sample._encoded = (self._featurizer, (featurized, scaled_runtime))
        return (featurized, scaled_runtime)
