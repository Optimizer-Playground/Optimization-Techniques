from __future__ import annotations

import collections
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, overload

import pandas as pd
import postbound as pb
import torch
from torch.utils.data import Dataset

from ._featurizer import BaoFeaturizer, FeaturizedNode


@dataclass
class BaoSample:
    plan: pb.QueryPlan
    runtime_ms: float

    def serialize(self) -> dict:
        return {"plan": pb.util.to_json(self.plan), "runtime": self.runtime_ms}


class BaoExperience(Dataset[tuple[torch.Tensor, torch.Tensor, float]]):
    @staticmethod
    def load(archive: Path | str, *, featurizer: BaoFeaturizer) -> BaoExperience:
        with open(archive, "r") as f:
            catalog = json.load(f)

        sample_df = pb.util.read_df(catalog["experience_archive"])
        samples: list[BaoSample] = []
        for row in sample_df.itertuples():
            plan = pb.opt.read_query_plan_json(row.plan)
            runtime = row.runtime
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

    def add(
        self, plan: pb.QueryPlan | BaoSample, runtime_ms: Optional[float] = None
    ) -> None:
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
        return self._new_samples >= self._retrain_freq

    def samples(self) -> Dataset[tuple[torch.Tensor, torch.Tensor, float]]:
        self._new_samples = 0
        return self

    @overload
    def sample(self, *, primitive: Literal[True]) -> Optional[tuple]: ...

    @overload
    def sample(self, *, primitive: Literal[False]) -> Optional[FeaturizedNode]: ...

    @overload
    def sample(self) -> Optional[FeaturizedNode]: ...

    def sample(self) -> Optional[FeaturizedNode]:
        if not self._storage:
            return None

        sample = self._storage[0]
        return self._featurizer.encode_plan(sample.plan)

    def clear(self) -> None:
        self._storage.clear()

    def store(
        self, catalog_path: Path | str, *, experience_path: Optional[Path | str] = None
    ) -> None:
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
        featurized = self._featurizer.encode_plan(sample.plan)
        scaled_runtime = self._featurizer.transform_runtime([(sample.runtime_ms,)])
        return (featurized, scaled_runtime)
