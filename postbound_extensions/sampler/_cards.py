from __future__ import annotations

import threading
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


class _ParallelLog:
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


class PostgresSamplerCtl:
    def __init__(
        self,
        sampler: Generator[pb.SqlQuery, None, None],
        out: Path,
        *,
        n_workers: int,
        pg_connect: str,
        timeout_ms: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.done = threading.Event()
        self.timeout_ms = timeout_ms

        self._critical_guard = threading.Lock()
        self._out = out
        self._log = _ParallelLog(verbose)

        self._pg_backends: dict[BackendHandle, psycopg.Connection] = {}
        self._pg_connect = pg_connect

        self._sampler = QuerySampler(sampler)
        self._n_generated = 0
        self._n_requested = 0

        self._n_workers = n_workers
        self._workers: list[threading.Thread] = []

    def sample(self, n_queries: int) -> None:
        self._n_generated = 0
        self._n_requested = n_queries
        self._log.restart(n_queries)
        for i in range(self._n_workers):
            worker = threading.Thread(
                target=sampling_worker, args=(self._sampler,), kwargs=dict(ctl=self)
            )
            self._workers.append(worker)
            worker.start()
        self.done.wait()

    def process_result(self, query: pb.SqlQuery, cardinality: pb.Cardinality) -> None:
        with self._critical_guard:
            self._n_generated += 1
            self._log.step()

            result_set = {"query": [query], "cardinality": [cardinality]}
            df = pd.DataFrame(result_set)
            pb.util.write_df(df, self._out, mode="a", header=not self._out.exists())

            if self._n_generated >= self._n_requested:
                self.done.set()
                self._close_backends_nolock()
                self._join_workers_nolock()

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

        # Sampling a new query can take quiet a bit of time, so we should re-check here if we
        # have been cancelled in the meantime!
        if ctl.done.is_set():
            break

        with conn.cursor() as cur:
            if ctl.timeout_ms is not None:
                cur.execute(f"SET LOCAL statement_timeout TO '{ctl.timeout_ms}ms'")  # type: ignore

            try:
                cur.execute(str(query))  # type: ignore
                result_set = cur.fetchone()
                if result_set is None:
                    continue
                cardinality = pb.Cardinality(result_set[0])
                ctl.process_result(query, cardinality)
            except psycopg.errors.QueryCanceled:
                # Timeout - do nothing, we just try again
                conn.rollback()
                continue
            except Exception:
                # Something went wrong and our connection might be bricked. We obtain a new one
                # just to be save
                handle, conn = ctl.obtain_pg_backend(handle)
                continue
