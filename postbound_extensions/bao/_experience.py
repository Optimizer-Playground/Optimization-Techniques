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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import postbound as pb
import torch
from torch.utils.data import Dataset

from ._featurizer import BaoFeaturizer, DatabaseCacheState


@dataclass
class BaoSample:
    """A single entry of the experience storage.

    Attributes / Parameters
    -------------------------
    plan : pb.QueryPlan
        The query plan that was executed
    runtime_ms : float
        The observed plan runtime in milliseconds
    """

    plan: pb.QueryPlan
    runtime_ms: float

    def serialize(self) -> dict:
        return {"plan": pb.util.to_json(self.plan), "runtime": self.runtime_ms}


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
        The featurizer is required to transform the query plans into feature vectors. Since each
        featurizer maintains a reference to the target database, we also use the featurizer to
        obtain the current cache state, should the need arise (see Warnings below).
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

    Warnings
    --------
    One current limitation of the experience store is the handling of the database cache state that
    is used for featurization. If the target database does not automatically add caching information
    to the query plans, the experience store tries to obtain the cache state once the samples are
    _retrieved_. Therefore, this data will likely be out-of-sync with the actual cache state at the
    time of query execution. It is generally best to only evaluate Bao on systems that provide full
    caching information in the query plans, such as Postgres.
    """

    @staticmethod
    def load(archive: Path | str, *, featurizer: BaoFeaturizer) -> BaoExperience:
        """Loads previously stored experience."""
        with open(archive, "r") as f:
            catalog = json.load(f)

        sample_df = pb.util.read_df(catalog["experience_archive"])
        samples: list[BaoSample] = []
        for row in sample_df.itertuples():
            plan = pb.opt.read_query_plan_json(row.plan)  # type: ignore
            runtime = row.runtime  # type: ignore
            samples.append(BaoSample(plan, runtime))

        return BaoExperience(
            featurizer,
            sample_window=catalog["window_size"],
            retraining_frequency=catalog["retrain_frequency"],
            existing_samples=samples,
        )

    @staticmethod
    def load_or_build(
        archive: Path | str, *, featurizer: BaoFeaturizer
    ) -> BaoExperience:
        """Loads experience from archive if it exists. Otherwise, creates a new experience."""
        archive = Path(archive)
        return (
            BaoExperience.load(archive, featurizer=featurizer)
            if archive.is_file()
            else BaoExperience(featurizer)
        )

    def __init__(
        self,
        featurizer: BaoFeaturizer,
        *,
        sample_window: int = 2000,
        retraining_frequency: int = 100,
        existing_samples: Optional[Iterable[BaoSample]] = None,
    ) -> None:
        self._featurizer = featurizer
        self._window = sample_window
        self._retrain_freq = retraining_frequency

        self._storage: collections.deque[BaoSample] = collections.deque(
            maxlen=self._window
        )
        if existing_samples is not None:
            self._storage.extend(existing_samples)

        self._new_samples = 0
        self._cache_state: Optional[DatabaseCacheState] = None

    def add(
        self, plan: pb.QueryPlan | BaoSample, runtime_ms: Optional[float] = None
    ) -> None:
        """Store a new sample in the experience.

        If the sample is given as a `BaoSample`, the runtime can be omitted. For plain query plans,
        it is required.
        """
        if isinstance(plan, BaoSample):
            self._storage.append(plan)
            self._new_samples += 1
            return

        if runtime_ms is None:
            raise ValueError(
                "runtime_ms is required if plan is supplied as a QueryPlan"
            )
        elif not isinstance(plan, pb.QueryPlan):
            raise ValueError("plan must be a QueryPlan")

        sample = BaoSample(plan, runtime_ms)
        self._storage.append(sample)
        self._new_samples += 1

    def should_retrain(self) -> bool:
        """Whether enough new samples have been added to justify retraining the TCNN."""
        return self._new_samples >= self._retrain_freq

    def samples(self) -> Dataset[tuple[torch.Tensor, torch.Tensor, float]]:
        """Provides the current experience as a PyTorch dataset of (features, runtimes) tuples."""
        self._cache_state = DatabaseCacheState(self._featurizer._db)
        return self

    def mark_retrained(self) -> None:
        """Marks the experience as having been used for retraining.

        This resets the internal new sample counter. This is called by the optimizer after
        retraining is completed. There is generally no need to call this method manually.
        """
        self._new_samples = 0
        self._cache_state = None

    def clear(self) -> None:
        """Removes all samples from the experience."""
        self._storage.clear()

    def store(
        self, catalog_path: Path | str, *, experience_path: Optional[Path | str] = None
    ) -> None:
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
        experience_path = experience_path or (
            catalog_path.parent / "experience.parquet"
        )
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

    def __getitem__(self, index):
        sample = self._storage[index]
        featurized = self._featurizer.encode_plan(
            sample.plan, cache_state=self._cache_state
        )
        scaled_runtime = self._featurizer.transform_runtime([(sample.runtime_ms,)])
        return (featurized, scaled_runtime)
