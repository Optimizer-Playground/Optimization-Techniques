from dataclasses import dataclass

import postbound as pb
from postbound.postgres import PostgresDatabase, PostgresPlan

from .._util import FgPbConverter
from ..hinting import HintSet


@dataclass(frozen=True)
class ResultSet:
    query: str
    hint_set_int: int
    hinted_query: str
    time: float
    timeout_used: float | None
    timed_out: bool
    explain_plan: PostgresPlan | None

    def __post_init__(self):
        if not isinstance(self.explain_plan, PostgresPlan) and self.explain_plan is not None:
            raise ValueError("Explain Plan is not a PostgresPlan")

def no_geqo(params: pb.PlanParameterization | None = None) -> pb.PlanParameterization:
    if params is None:
        params = pb.PlanParameterization()
    params.set_system_settings(geqo="off")
    return params

def no_para(params: pb.PlanParameterization | None = None) -> pb.PlanParameterization:
    if params is None:
        params = pb.PlanParameterization()
    params.set_system_settings(max_parallel_workers_per_gather=0)
    return params


class QueryExecutor:
    def __init__(self, db_conn: PostgresDatabase, plan_params: pb.PlanParameterization | None):
        self.db = db_conn
        self.converter = FgPbConverter()
        self._plan_params = plan_params

    def get_backend_id(self) -> int:
        result = self.db.execute_query("SELECT pg_backend_pid();", raw=True)
        if not result:
            raise RuntimeError("Could not retrieve backend PID")
        return int(result[0][0])

    def execute_query(
            self,
            query_str: str,
            hint_set: HintSet,
            timeout_s: float | None = None,
            explain: bool = False,
            explain_analyze: bool = False,
            *,
            plan_parameters: pb.PlanParameterization | None = None,
    ) -> ResultSet:

        plan_params = self._plan_params if self._plan_params is not None else None
        plan_params = plan_parameters if plan_parameters is not None else plan_params

        hinted_query = self.db.hinting().generate_hints(
            query=pb.parse_query(query_str),
            physical_operators=self.converter.fg_to_pb_hint_set(hint_set),
            plan_parameters=plan_params,
        )

        if explain:
            hinted_query = pb.transform.as_explain(hinted_query)
        elif explain_analyze:
            hinted_query = pb.transform.as_explain_analyze(hinted_query)
        else:
            pass

        try:
            result = self.db.execute_query(
                hinted_query,
                timeout=timeout_s if timeout_s is not None else None
            )
            result_time_s = self.db.last_query_runtime()
            writeable_result = PostgresPlan(result) if explain or explain_analyze else None
            timed_out = False
        except TimeoutError:
            result_time_s = timeout_s if timeout_s is not None else self.db.last_query_runtime()
            writeable_result = None
            timed_out = True

        query_result = ResultSet(
            query=query_str,
            hint_set_int=hint_set.int_rep,
            hinted_query=str(hinted_query),
            time=result_time_s,
            timeout_used=timeout_s,
            timed_out=timed_out,
            explain_plan=writeable_result,
        )

        return query_result
