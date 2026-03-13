from dataclasses import dataclass
from typing import Optional

from postbound.postgres import PostgresExplainPlan

from .._util import FgPbConverter
from ..hinting import HintSet
import postbound as pb


@dataclass(frozen=True)
class ResultSet:
    query: str
    hint_set_int: int
    hinted_query: str
    time: float
    timeout_used: float
    timed_out: bool
    explain_plan: Optional[PostgresExplainPlan]

    def __post_init__(self):
        if not isinstance(self.explain_plan, PostgresExplainPlan) and self.explain_plan is not None:
            raise ValueError("Explain Plan is not a PostgresExplainPlan")

def no_geqo(params: Optional[pb.PlanParameterization] = None) -> pb.PlanParameterization:
    if params is None:
        params = pb.PlanParameterization()
    params.set_system_settings(geqo="off")
    return params

def no_para(params: Optional[pb.PlanParameterization] = None) -> pb.PlanParameterization:
    if params is None:
        params = pb.PlanParameterization()
    params.set_system_settings(max_parallel_workers_per_gather=0)
    return params


class QueryExecutor:
    def __init__(self, db_conn: pb.postgres.PostgresInterface, plan_params: pb.PlanParameterization):
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
            timeout_s: Optional[float] = None,
            explain: bool = False,
            explain_analyze: bool = False,
            *,
            plan_parameters: Optional[pb.PlanParameterization] = None,
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
            writeable_result = PostgresExplainPlan(result[0])
        except TimeoutError:
            result = None
            result_time_s = timeout_s
            writeable_result = None

        query_result = ResultSet(
            query=query_str,
            hint_set_int=hint_set.int_rep,
            hinted_query=str(hinted_query),
            time=result_time_s,
            timeout_used=timeout_s,
            timed_out=(result is None),
            explain_plan=writeable_result,
        )

        return query_result