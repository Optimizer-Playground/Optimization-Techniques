from __future__ import annotations

import queue
import threading
from collections.abc import Generator, Sequence
from concurrent import futures
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import postbound as pb
import psycopg
from tqdm import tqdm

type Handle = int
type Status = Literal["ok", "timeout", "error"]


class ParallelLog:
    def __init__(self, enabled: bool = True) -> None:
        self._n_queries = 0
        self._enabled = enabled
        self._logger = pb.util.standard_logger(self._enabled)
        self._critical_guard = threading.Semaphore()

        self._progress_bar = None

    @property
    def n_queries(self) -> int:
        return self._n_queries

    @n_queries.setter
    def n_queries(self, n_queries: int) -> None:
        self._n_queries = n_queries
        self._progress_bar = (
            tqdm(total=n_queries, desc="Sample", unit="q") if self._enabled else None
        )

    def print(self, *args) -> None:
        return self(*args)

    def log(self, *args) -> None:
        return self(*args)

    def sample_acquired(self) -> None:
        if not self._enabled or self._progress_bar is None:
            return

        with self._critical_guard:
            self._progress_bar.update()

    def __call__(self, *args) -> None:
        if not self._enabled:
            return
        prefix = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thread = threading.current_thread()
        with self._critical_guard:
            self._logger(f"[{prefix}] [{thread.name}]", *args)


@dataclass
class CardinalitySample:
    query: pb.SqlQuery
    plan: pb.QueryPlan
    cardinality: pb.Cardinality

    @staticmethod
    def csv_cols() -> Sequence[str]:
        return ["query", "plan", "cardinality"]

    def __json__(self) -> pb.util.jsondict:
        return asdict(self)


class ConnectionPool:
    def __init__(
        self, connect_string: str, n_connections: int, log: ParallelLog
    ) -> None:
        self._connect_string = connect_string
        self._log = log

        self._critical_guard = threading.Semaphore()
        self._connection_guard = threading.Semaphore(n_connections)
        self._idle_connections: dict[Handle, psycopg.Connection] = {}
        self._busy_connections: dict[Handle, psycopg.Connection] = {}
        self._backend_pids: dict[Handle, int] = {}

        for i in range(n_connections):
            conn = psycopg.connect(self._connect_string)
            self._backend_pids[i] = conn.info.backend_pid
            self._idle_connections[i] = conn

        self._connection_watchdog = psycopg.connect(self._connect_string)
        self._shut_down = threading.Event()

    def acquire(self) -> tuple[Handle, psycopg.Connection]:
        if self._shut_down.is_set():
            raise RuntimeError("Connection pool has been shut down")

        self._connection_guard.acquire()  # wait until a connection is available

        with self._critical_guard:
            handle, conn = self._idle_connections.popitem()
            self._busy_connections[handle] = conn

        return handle, conn

    def release(self, handle: Handle) -> None:
        if self._shut_down.is_set():
            return

        with self._critical_guard:
            conn = self._busy_connections.pop(handle, None)
            if conn is None:
                # the pool could have been shut down by the main process in the meantime
                return
            conn.rollback()
            self._idle_connections[handle] = conn

        self._connection_guard.release()  # signal that a new connection is available

    def reset(self, handle: Handle) -> None:
        if self._shut_down.is_set():
            return

        new_conn = psycopg.connect(self._connect_string)

        with self._critical_guard:
            current_conn = self._busy_connections.pop(handle, None)
            if current_conn is None:
                # the pool could have been shut down by the main process in the meantime
                return

            try:
                current_conn.close()
            except Exception:
                pass

            self._idle_connections[handle] = new_conn
            self._backend_pids[handle] = new_conn.info.backend_pid

        self._connection_guard.release()  # signal that a new connection is available

    def shutdown(self, force: bool = False) -> None:
        self._shut_down.set()

        with self._critical_guard:
            for conn in self._idle_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._idle_connections.clear()

            if not force:
                # make sure to keep this in sync with the normal exit
                self._backend_pids.clear()
                self._connection_watchdog.close()
                return

            for handle, conn in self._busy_connections.items():
                self._cancel_connection(handle)
                self._connection_guard.release()
                try:
                    conn.cancel_safe()
                    conn.close()
                except Exception:
                    pass
            self._busy_connections.clear()

            # make sure to keep this in sync with the early exit
            self._backend_pids.clear()
            self._connection_watchdog.close()

    def _cancel_connection(self, handle: Handle) -> None:
        backend_pid = self._backend_pids[handle]
        try:
            with self._connection_watchdog.cursor() as cur:
                cur.execute("SELECT pg_cancel_backend(%s)", (backend_pid,))
        except Exception:
            self._connection_watchdog.rollback()


@dataclass
class SamplingCtx:
    queries: queue.Queue[pb.SqlQuery]
    results: queue.Queue[CardinalitySample | None]
    connections: ConnectionPool
    done: threading.Event


def execute_query(
    query: pb.SqlQuery,
    *,
    connection_pool: ConnectionPool,
    log: ParallelLog,
    timeout_ms: Optional[float] = None,
) -> tuple[Status, CardinalitySample | None]:
    handle, conn = connection_pool.acquire()
    prepared = pb.transform.as_explain_analyze(query)

    with conn.cursor() as cursor:
        if timeout_ms:
            cursor.execute(f"SET statement_timeout TO '{timeout_ms}ms'")  # type: ignore[arg-type]

        try:
            cursor.execute(str(prepared))  # type: ignore[arg-type]
            raw_plan = cursor.fetchone()[0]  # type: ignore[index] - if this errors we end up in the except block
            parsed = pb.postgres.PostgresExplainPlan(raw_plan).as_qep()
            cardinality = parsed.actual_cardinality
            sample = CardinalitySample(query, parsed, cardinality)
            connection_pool.release(handle)
            return "ok", sample
        except psycopg.errors.QueryCanceled:
            connection_pool.release(handle)
            return "timeout", None
        except Exception as e:
            thread = threading.current_thread()
            log(
                thread.name,
                ":: Query",
                query,
                "produced error",
                type(e).__name__,
                "-",
                e,
            )
            connection_pool.reset(handle)
            return "error", None


class CardinalitySampler:
    def __init__(
        self,
        generator: Generator[pb.SqlQuery, None, None],
        *,
        connect_string: str,
        n_workers: int,
        timeout_ms: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self._generator = generator
        self._timeout_ms = timeout_ms
        self._n_workers = n_workers

        self._verbose = verbose
        self._log = ParallelLog(self._verbose)

        self._connection_pool = ConnectionPool(
            connect_string, n_connections=self._n_workers, log=self._log
        )
        self._done = threading.Event()

        self._worker_pool = futures.ThreadPoolExecutor(
            max_workers=self._n_workers, thread_name_prefix="CardinalitySampler"
        )
        self._critical_guard = threading.Semaphore()

        self._n_samples = 0
        self._sampled_queries: set[pb.SqlQuery] = set()
        self._samples: list[CardinalitySample] = []
        self._out_file: Path | None = None

    def sample(
        self, n_queries: int, *, stream_to: Optional[Path] = None
    ) -> list[CardinalitySample]:
        self._n_queries = n_queries
        self._log.n_queries = n_queries
        self._out_file = stream_to
        for query in range(self._n_queries):
            self._enqueue_sampler()

        try:
            self._done.wait()
        except KeyboardInterrupt:
            pass
        return self._samples

    def shutdown(self) -> None:
        self._done.set()
        # For shutdown, we need to close the worker pool before shutting down the connections.
        # The reason is that the worker pool first tries to acquire() a connection from the pool
        # If the connection pool is already shut down, acquire() hangs indefinitely because no
        # more connections are available. By shutting down the worker pool first, active workers
        # might run slightly longer until the connection is closed and execution fails. For now,
        # this is acceptable.
        self._worker_pool.shutdown(wait=False, cancel_futures=True)
        self._connection_pool.shutdown(force=True)

    def _worker_callback(self, future: futures.Future) -> None:
        if future.cancelled():
            return

        status, sample = future.result()

        match status:
            case "ok":
                self._process_sample(sample)
            case "timeout" | "error" if not self._done.is_set():
                # we still need more samples, so create another one
                self._enqueue_sampler()
            case "timeout" | "error" if self._done.is_set():
                # we did not obtain a valid sample, but we have enough data already
                # no need to sample a new query
                pass

    def _process_sample(self, sample: CardinalitySample) -> None:
        with self._critical_guard:
            self._samples.append(sample)
            self._stream_result(sample)
            n_acquired = len(self._samples)

        if n_acquired >= self._n_queries:
            self._done.set()

        self._log.sample_acquired()

    def _enqueue_sampler(self) -> None:
        if self._done.is_set():
            return

        with self._critical_guard:
            while (query := next(self._generator)) in self._sampled_queries:
                continue

        if self._done.is_set():
            return
        future = self._worker_pool.submit(
            execute_query,
            query,
            connection_pool=self._connection_pool,
            log=self._log,
            timeout_ms=self._timeout_ms,
        )
        future.add_done_callback(self._worker_callback)

    def _stream_result(self, sample: CardinalitySample) -> None:
        if self._out_file is None:
            return

        serialized = {key: pb.util.to_json(val) for key, val in asdict(sample).items()}
        df = pd.DataFrame([serialized])
        df.to_csv(self._out_file, mode="a", index=False, header=False)
