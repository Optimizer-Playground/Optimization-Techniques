
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Sequence, Collection

import pandas as pd
import postbound as pb
from bidict import bidict
from postbound.postgres import PostgresExplainPlan
from tqdm import tqdm

from ._executor import ResultSet, QueryExecutor, no_geqo
from ._result import LabelingResult
from ._util import get_neighbors, HintSetStats
from .._util import prepare_dir
from ..hinting import CORE_HINT_LIBRARY
from ..hinting import HintSet, HintSetFactory


class FastgresLabelProvider:

    def __init__(self, label_path: Path | str):
        self._label_path = label_path
        if isinstance(self._label_path, str):
            self._label_path = Path(self._label_path)
        self._df = pd.read_csv(self._label_path)
        self._dict = dict()
        self._build_label_dict()

    @property
    def size(self) -> int:
        return len(self._dict.keys())

    def _build_label_dict(self):
        filtered = self._df[(~self._df["timeout"]) & (self._df["opt"])]
        for query_name, hint_set_int in filtered[["query_name", "hint_set_int"]].itertuples(index=False, name=None):
            self._dict[query_name] = hint_set_int
            if query_name.endswith(".sql"):
                self._dict[query_name.rstrip(".sql")] = hint_set_int


    def get_label(self, query_name: str) -> int:
        return self._dict[query_name]

    def get_labels(self, query_names: Sequence[str]) -> Collection[int]:
        labels = list()
        for query_name in query_names:
            labels.append(self._dict[query_name])
        return labels


ExperienceStore = dict[int, HintSetStats]

def update_experience(hint_set: int, is_better: bool, store: ExperienceStore):
    if hint_set not in store:
        store[hint_set] = HintSetStats()
    store[hint_set].update(is_better)

@dataclass
class FastLabelSettings:
    enable_experience: bool = True
    enable_early_stopping: bool = True
    enable_level_cap: bool  = True

    baseline_timeout: float = 150.0  # s
    level_cap: int = 4
    early_stopping_threshold: int = 2
    absolute_timeout: float = 0.5  # s
    relative_timeout: float = 1.2  # factor

class WorkloadLabelSettings:

    def __init__(self, query_path: Path | str, db_string: str, *, label_settings: FastLabelSettings):
        if isinstance(query_path, str):
            query_path = Path(query_path)
        self._query_path = query_path
        self._db_string = db_string

        self.fls = label_settings

        self.workload = pb.Workload.read(self._query_path)
        self.dbc = pb.postgres.connect(application_name="FastLabel", connect_string=db_string)
        self.plan_params = no_geqo()
        self.qex = QueryExecutor(self.dbc, plan_params=no_geqo())
        self.hsf = HintSetFactory(CORE_HINT_LIBRARY)
        self.hint_size = self.hsf.hint_library.size

        self.exp = ExperienceStore()

    def get_timeout(self, current_timeout: float):
        return max(self.fls.absolute_timeout, current_timeout * self.fls.relative_timeout)


class QueryLabeling:

    def __init__(self, query_name: str, settings: WorkloadLabelSettings):

        # These are labeling invariant
        self.q_name = query_name
        self.settings = settings
        self.q_string = str(self.settings.workload[self.q_name])
        self.exec = self.settings.qex.execute_query
        self.hsf = self.settings.hsf
        self.result_list = list() # result gathering

        # These are modified during labeling
        self.restrictions = set()
        self.seen_plans = dict()
        self.current_level = 0
        self.best_candidates = bidict()
        self.timeout = self.settings.fls.baseline_timeout
        self.optimal_candidate: Optional[ResultSet] = None
        self.current_optimal_time = self.settings.fls.baseline_timeout


    def break_level(self):
        return self.current_level >= self.settings.fls.level_cap

    def check_plan(self, hs: HintSet) -> tuple[Optional[ResultSet], PostgresExplainPlan]:
        res: ResultSet = self.exec(self.q_string, hs, self.settings.fls.baseline_timeout, explain=True)
        return self.seen_plans.get(res.explain_plan, None), res.explain_plan

    def eval_hs(self, hs: HintSet, timeout: float) -> tuple[ResultSet, bool]:
        # explain suffices
        found_result, explain_plan = self.check_plan(hs)  # this should always have a plan
        if found_result is not None:
            return_result = ResultSet(
                query=self.q_string,
                hint_set_int=hs.hs_int,
                hinted_query=found_result.hinted_query,
                time=found_result.time,
                timeout_used=found_result.timeout_used,
                timed_out=found_result.timed_out,
                explain_plan=found_result.explain_plan,
            )
            return return_result, True

        res: ResultSet = self.exec(self.q_string, hs, timeout, explain_analyze=True)
        if res.explain_plan is None:
            res = ResultSet(
                query=res.query,
                hint_set_int=res.hint_set_int,
                hinted_query=res.hinted_query,
                time=res.time,
                timeout_used=res.timeout_used,
                timed_out=res.timed_out,
                explain_plan=explain_plan
            )

        self.seen_plans[explain_plan] = res
        return res, False

    def check_optimality(self, candidate: ResultSet) -> None:
        if self.optimal_candidate is None or candidate.time < self.optimal_candidate.time:
            self.optimal_candidate = candidate

    def get_new_candidates(self, hs_i: int) -> list[HintSet]:
        neighbors = get_neighbors(hs_i, self.settings.exp, self.restrictions)
        return [self.hsf.hint_set(_) for _ in neighbors]

    def adjust_timeout(self, result: ResultSet) -> None:
        if result.time < self.current_optimal_time:
            self.current_optimal_time = result.time
            self.timeout = self.current_optimal_time

    def check_es(self) -> bool:
        optimal_candidate_level = self.best_candidates.inverse[self.optimal_candidate]
        return abs(optimal_candidate_level - self.current_level) >= self.settings.fls.early_stopping_threshold

    def update_experience(self, result: ResultSet) -> None:
        was_better = result.time < self.best_candidates[0].time
        update_experience(result.hint_set_int, was_better, self.settings.exp)

    def update_best_neighbor(self, best_candidate: ResultSet):
        if self.current_level not in self.best_candidates:
            if best_candidate in self.best_candidates.inv:
                pass
            else:
                self.best_candidates[self.current_level] = best_candidate

    def prepare_results(self) -> list[LabelingResult]:
        writable = list()
        for query_result, seen in self.result_list:
            hs = self.hsf.hint_set(query_result.hint_set_int)
            is_opt = True if self.optimal_candidate == query_result else False
            chosen = True if query_result in self.best_candidates.inverse else False
            writable_result = LabelingResult(
                query_name=self.q_name.rstrip(".sql"),
                hint_set_int=query_result.hint_set_int,
                binary_rep=hs.bin_rep,
                measured_time=query_result.time,
                occurred_level=hs.n_disabled,
                is_opt=is_opt,
                had_timeout=query_result.timed_out,
                chosen_in_level=chosen,
                removed=False,
                seen_plan=seen,
                hint_names=self.hsf.hint_names,
                qep=query_result.explain_plan.explain_data
            )
            writable.append(writable_result)
        return writable

    def label_query(self):
        queue = deque([self.hsf.default_hint_set()])
        while queue:
            if self.break_level():
                break
            neighbors = list()
            for _ in range(len(queue)):
                current_hs = queue.popleft()
                result, seen = self.eval_hs(current_hs, self.timeout)
                if current_hs.n_disabled == 0:
                    self.best_candidates[current_hs.n_disabled] = result
                neighbors.append(result)
                self.result_list.append((result, seen))
                self.update_experience(result)
                self.adjust_timeout(result)
            best_candidate = sorted(neighbors, key=lambda x: x.time)[0]
            self.update_best_neighbor(best_candidate)
            self.check_optimality(best_candidate)
            if self.check_es():
                break
            new_candidates = self.get_new_candidates(best_candidate.hint_set_int)
            queue.extend(new_candidates)
            self.current_level += 1
        return self.prepare_results()


class WorkloadLabeling:

    def __init__(self, wl_settings: WorkloadLabelSettings):
        self.experiment_settings = wl_settings

    def label_queries(self) -> Iterable[pd.DataFrame]:
        all_labels = list()
        wl_q = self.experiment_settings.workload.queries()
        for query_index, query_object in tqdm(enumerate(wl_q), desc="Labeling Queries", total=len(wl_q)):
            query_name = self.experiment_settings.workload.label_of(query_object)
            query_labeling = QueryLabeling(
                query_name=query_name,
                settings=self.experiment_settings,
            )
            query_labels = query_labeling.label_query()
            all_labels.extend(query_labels)
            yield pd.DataFrame([result.to_dict() for result in all_labels])

    def label_and_save_queries(self, save_dir: Path) -> None:
        prepare_dir(save_dir)
        for intermediate_df in self.label_queries():
            intermediate_df.to_csv(save_dir / "label.csv", index=False)