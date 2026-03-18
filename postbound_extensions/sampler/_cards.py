from __future__ import annotations

import threading
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import pandas as pd
import postbound as pb
import psycopg
from tqdm import tqdm

type BackendHandle = int


class QuerySampler:
    def __init__(self, generator: Generator[pb.SqlQuery, None, None]) -> None:
        self._critical_guard = threading.Lock()
        self._generator = generator
        self._generated_queries: set[pb.SqlQuery] = set()

    def next_query(self) -> pb.SqlQuery:
        with self._critical_guard:
            while (query := next(self._generator)) in self._generated_queries:
                continue
            self._generated_queries.add(query)
        return query


class ParallelLog:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._critical_guard = threading.Lock()
        self._tqdm = None

    def restart(self, n_queries: int) -> None:
        if not self._enabled:
            return
        if self._tqdm is not None:
            self._tqdm.close()
        self._tqdm = tqdm(total=n_queries, unit="q", leave=True)

    def step(self) -> None:
        if not self._enabled:
            return
        assert self._tqdm is not None
        self._tqdm.update()

    def __call__(self, *args) -> None:
        if not self._enabled:
            return
        assert self._tqdm is not None

        msg = " ".join(str(arg) for arg in args)
        thread_name = threading.current_thread().name
        thread_id = threading.get_ident()
        with self._critical_guard:
            self._tqdm.write(f"[{thread_name}-{thread_id}] {msg}")


class PostgresSamplerCtl:
    def __init__(
        self,
        sampler: Generator[pb.SqlQuery, None, None],
        out: Path,
        *,
        n_workers: int,
        pg_connect: str,
        allow_zero_tuples: bool = False,
        timeout_ms: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.done = threading.Event()
        self.timeout_ms = timeout_ms
        self.allow_zero_tuples = allow_zero_tuples

        self._critical_guard = threading.Lock()
        self._out = out
        self._log = ParallelLog(verbose)

        self._pg_backends: dict[BackendHandle, psycopg.Connection] = {}
        self._pg_connect = pg_connect

        self._sampler = QuerySampler(sampler)
        self._n_generated = 0
        self._n_requested = 0

        self._n_workers = n_workers
        self._workers: list[threading.Thread] = []

    @property
    def log(self) -> ParallelLog:
        return self._log

    def sample(self, n_queries: int) -> None:
        self.done.clear()
        self._n_generated = 0
        self._n_requested = n_queries
        self._log.restart(n_queries)

        for i in range(self._n_workers):
            worker = threading.Thread(
                target=sampling_worker,
                args=(self._sampler,),
                kwargs=dict(ctl=self),
                name="SamplingWorker",
            )
            self._workers.append(worker)
            worker.start()

        self.done.wait()
        self._close_backends_nolock()
        self._join_workers_nolock()

    def process_result(
        self, query: pb.SqlQuery, plan: pb.postgres.PostgresExplainPlan
    ) -> None:
        with self._critical_guard:
            if plan.root.true_cardinality == 0 and not self.allow_zero_tuples:
                return

            result_set = {
                "query": [query],
                "plan": [plan],
                "cardinality": [plan.root.true_cardinality],
                "runtime_ms": [plan.root.execution_time],
            }
            df = pd.DataFrame(result_set)
            pb.util.write_df(df, self._out, mode="a", header=not self._out.exists())

            self._n_generated += 1
            self._log.step()

            if self._n_generated >= self._n_requested:
                self.done.set()

    def obtain_pg_backend(
        self, prev_handle: Optional[BackendHandle] = None
    ) -> tuple[BackendHandle, psycopg.Connection]:
        if self.done.is_set():
            return -1, None  # type: ignore

        with self._critical_guard:
            conn = psycopg.connect(self._pg_connect)
            handle = conn.info.backend_pid
            self._pg_backends[handle] = conn
            if prev_handle is None:
                return handle, conn

            prev_conn = self._pg_backends.pop(prev_handle, None)
            if prev_conn is None:
                return handle, conn

            try:
                prev_conn.close()
            except Exception:
                pass

        return handle, conn

    def shutdown(self) -> None:
        self.done.set()
        with self._critical_guard:
            self._close_backends_nolock()
            self._join_workers_nolock()

    def _join_workers_nolock(self) -> None:
        for worker in self._workers:
            worker.join()
        self._workers.clear()

    def _close_backends_nolock(self) -> None:
        with psycopg.connect(self._pg_connect) as conn:
            cur = conn.cursor()
            for backend in self._pg_backends:
                cur.execute("SELECT pg_cancel_backend(%s)", (backend,))
            self._pg_backends.clear()


def sampling_worker(query_sampler: QuerySampler, *, ctl: PostgresSamplerCtl) -> None:
    handle, conn = ctl.obtain_pg_backend()
    while not ctl.done.is_set():
        query = query_sampler.next_query()
        query = pb.transform.as_star_query(query)
        explain_query = pb.transform.as_explain_analyze(query)

        # Sampling a new query can take quiet a bit of time, so we should re-check here if we
        # have been cancelled in the meantime!
        if ctl.done.is_set():
            break

        with conn.cursor() as cur:
            if ctl.timeout_ms is not None:
                cur.execute(f"SET LOCAL statement_timeout TO '{ctl.timeout_ms}ms'")  # type: ignore

            try:
                cur.execute(str(explain_query))  # type: ignore
                result_set = cur.fetchone()
                if result_set is None:
                    continue
                ctl.process_result(
                    query, pb.postgres.PostgresExplainPlan(result_set[0])
                )
            except psycopg.errors.QueryCanceled:
                # Timeout - do nothing, we just try again
                conn.rollback()
                continue
            except Exception as e:
                # Something went wrong and our connection might be bricked. We obtain a new one
                # just to be save
                stack_trace = traceback.format_exc()
                ctl.log("Error:", e, "// Stack trace:", stack_trace, "\n")
                handle, conn = ctl.obtain_pg_backend(handle)
                continue
