"""Public FASTgres workflow with prepared contexts and portable model bundles."""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import postbound as pb

from ._util import FgPbConverter, prepare_dir
from .context import ContextFactory, ContextManager, CtxGranularity, DatabaseSchema
from .featurization import DatabaseConnection, EncodingInformation, FastgresFeaturization
from .hinting import CORE_HINT_LIBRARY, Hint, HintLibrary, HintSet, HintSetFactory
from .labeling import FastgresLabelProvider, FastLabelSettings, WorkloadLabeling, WorkloadLabelSettings
from .model import FastgresContextModel, FastgresModel

type LabelSource = FastgresLabelProvider | pd.DataFrame | Mapping[str, int] | Path | str


class FastgresContextWarning(UserWarning):
    """A query has no covered, trained context and receives PostgreSQL defaults."""


class FastgresOptimizer(pb.CompleteOptimizationAlgorithm, pb.OperatorSelection):
    """Coordinate preparation, FASTlabel, training, prediction, and persistence.

    The default SUPERTABLE mode uses one model and a fixed feature layout over
    the preparation workload's tables. Training subsets and new queries resolve
    against that same layout. Uncovered contexts warn and return the defaults
    declared by ``hint_library``; other errors are not converted into fallbacks.

    Construction does not query a database or start training. Call ``prepare``
    explicitly, or inject a prepared featurizer and optionally a trained model.
    Advanced experiments can continue using the component classes independently.

    Models loaded by ``pre_trained`` need neither training labels nor a database
    for prediction. A database is required for labeling and PostBOUND integration.
    The latter currently supports CORE_HINT_LIBRARY, as does the FASTlabel bridge.
    """

    _FORMAT_VERSION = 1
    _FEATURE_FORMAT = "fastgres-four-values-per-column-v1"

    def __init__(
        self,
        *,
        database: pb.postgres.PostgresDatabase | None = None,
        featurizer: FastgresFeaturization | None = None,
        model: FastgresContextModel | None = None,
        context_granularity: CtxGranularity = CtxGranularity.SUPERTABLE,
        hint_library: HintLibrary = CORE_HINT_LIBRARY,
        model_params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not hint_library.size:
            raise ValueError("A FASTgres hint library must not be empty")
        if any(type(h.db_instr_val) is not bool for h in hint_library.hint_list):
            raise ValueError("Hint defaults must be booleans")
        self._database = database
        self._hint_library = HintLibrary(list(hint_library.hint_list))
        self._factory = HintSetFactory(self._hint_library)
        self._granularity = context_granularity
        self._model_params = {"n_estimators": 100, "max_depth": 1000, **(model_params or {})}
        self._featurizer: FastgresFeaturization | None = None
        self._model: FastgresContextModel | None = None
        self._training_metrics: dict = {}
        if model is not None:
            if featurizer is not None and model.featurizer is not featurizer:
                raise ValueError("An injected model must use the injected featurizer")
            featurizer = model.featurizer
            if model_params is None:
                self._model_params = dict(model.gb_kwargs)
        if featurizer is not None:
            source = featurizer.ctx_manager
            schema = copy.deepcopy(source.db_schema)
            assert isinstance(schema, DatabaseSchema)
            manager = ContextManager.from_contexts(source.ctx_granularity, copy.deepcopy(list(source.ctx2q)), schema)
            self._granularity = source.ctx_granularity
            self._featurizer = FastgresFeaturization(
                None, manager, enc_info=featurizer.statistics, db_schema=schema)
        if model is not None:
            # Rebind the coordinator without mutating the supplied research components.
            assert self._featurizer is not None
            self._model = copy.copy(model)
            self._model.cm = self._featurizer.ctx_manager
            self._model.featurizer = self._featurizer
            self._model.label_provider = None
            self._validate_models()

    @property
    def featurizer(self) -> FastgresFeaturization:
        if self._featurizer is None:
            raise ValueError("Prepare the FASTgres featurizer before training or prediction")
        return self._featurizer

    @property
    def context_manager(self) -> ContextManager:
        return self.featurizer.ctx_manager

    @property
    def model(self) -> FastgresContextModel:
        if self._model is None or not self._model.models:
            raise ValueError("Train or load a FASTgres model before prediction")
        return self._model

    @property
    def hint_library(self) -> HintLibrary:
        return HintLibrary(list(self._hint_library.hint_list))

    @property
    def training_metrics(self) -> dict:
        return dict(self._training_metrics)

    def _require_database(self) -> pb.postgres.PostgresDatabase:
        if self._database is None:
            raise ValueError("This operation requires a PostgreSQL database connection")
        return self._database

    def _require_core_hints(self) -> None:
        if self._hint_library != CORE_HINT_LIBRARY:
            raise ValueError("PostBOUND operator selection and FASTlabel currently support the six core hints only")

    def prepare(
        self,
        workload: pb.Workload,
        *,
        encodings: EncodingInformation | Path | str | None = None,
        schema: DatabaseSchema | None = None,
        skipped_columns: dict | None = None,
    ) -> FastgresOptimizer:
        """Establish contexts and encodings once; collect statistics only if needed.

        ``workload`` defines the feature universe and can include evaluation
        queries. Training can subsequently use a subset. Provide cached encodings
        and a schema to prepare without a database. To change the feature universe,
        prepare and train a new optimizer instead of modifying an existing model.
        """
        if self._featurizer is not None:
            raise ValueError("Contexts are already prepared; use a new optimizer to change the feature layout")
        if not workload:
            raise ValueError("Cannot prepare FASTgres with an empty workload")
        schema = copy.deepcopy(schema) if schema is not None else DatabaseSchema.from_dbc(self._require_database())
        manager = ContextManager(self._granularity, workload.queries(), schema, fixed_contexts=True)
        manager.classify_queries(workload.queries())  # Validate binding and coverage before expensive collection.
        if isinstance(encodings, (str, Path)):
            encodings = EncodingInformation.load_encoding_info(Path(encodings))
        if encodings is None:
            encodings = EncodingInformation.build(
                DatabaseConnection(self._require_database()), workload, skipped_columns or {})
        elif skipped_columns is not None and skipped_columns != encodings.skipped_columns:
            raise ValueError("Skipped columns must agree with the supplied encoding statistics")
        self._featurizer = FastgresFeaturization(None, manager, enc_info=encodings, db_schema=schema)
        return self

    def label(self, workload: pb.Workload, *, settings: FastLabelSettings | None = None) -> pd.DataFrame:
        """Explicitly run FASTlabel; return a DataFrame accepted by ``train``."""
        if not workload:
            raise ValueError("Cannot label an empty workload")
        self._require_core_hints()
        configuration = WorkloadLabelSettings(
            workload=workload, database=self._require_database(), label_settings=settings or FastLabelSettings())
        result = pd.DataFrame()
        for result in WorkloadLabeling(configuration).label_queries():
            pass  # Each yielded frame contains all labels collected so far.
        return result

    @staticmethod
    def _labels(source: LabelSource) -> FastgresLabelProvider:
        if isinstance(source, FastgresLabelProvider):
            return source
        if isinstance(source, Mapping):
            source = pd.DataFrame({"query_name": list(source), "hint_set_int": list(source.values()),
                                   "opt": True, "timeout": False})
        return FastgresLabelProvider(source)

    def train(self, workload: pb.Workload, *, labels: LabelSource) -> dict:
        """Train on supplied labels using the prepared context and feature layout.

        Training never runs labeling or silently expands the prepared schema.
        A successful call replaces the previous models; a failed call keeps them.
        """
        if not workload:
            raise ValueError("Cannot train FASTgres with an empty workload")
        manager = self.context_manager
        manager.classify_queries(workload.queries())
        provider = self._labels(labels)
        values = provider.get_labels([workload.label_of(q) for q in workload.queries()])
        if values.dtype.kind not in "biu" or any(not 0 <= int(v) < 2 ** self._hint_library.size for v in values):
            raise ValueError("Training labels must be integer hint sets belonging to the configured library")
        model = FastgresContextModel(self.featurizer, provider, manager, **self._model_params)
        model.fit(workload)
        self._model = model
        self._training_metrics = {"status": "completed", "queries": len(workload), "contexts": len(model.models)}
        return self.training_metrics

    def fit(self, workload: pb.Workload, *, labels: LabelSource) -> dict:
        return self.train(workload, labels=labels)

    def predict(self, query: pb.SqlQuery) -> HintSet:
        """Predict one hint set; uncovered or untrained contexts warn and use PG defaults."""
        model = self.model  # Initialization errors must not look like successful fallbacks.
        context = self.context_manager.resolve(query)
        if context is None or context not in model.models:
            warnings.warn("No trained FASTgres context covers this query; returning PostgreSQL defaults "
                          "declared by the hint library.", FastgresContextWarning, stacklevel=2)
            return self._factory.database_default_hint_set()
        return self._factory.hint_set(model.predict(query))

    def predict_queries(self, queries: Iterable[pb.SqlQuery] | pb.Workload) -> list[HintSet]:
        return [self.predict(q) for q in (queries.queries() if isinstance(queries, pb.Workload) else queries)]

    def select_physical_operators(self, query: pb.SqlQuery,
                                  join_order: pb.JoinTree | None = None) -> pb.PhysicalOperatorAssignment:
        self._require_core_hints()
        return FgPbConverter().fg_to_pb_hint_set(self.predict(query))

    def optimize_query(self, query: pb.SqlQuery) -> pb.QueryPlan:
        database = self._require_database()
        operators = self.select_physical_operators(query)
        hinted = database.hinting().generate_hints(query, physical_operators=operators)
        return database.optimizer().query_plan(hinted)

    def _validate_models(self) -> None:
        for context, model in self.model.models.items():
            if context not in self.context_manager.ctx2q:
                raise ValueError("Model context does not belong to the prepared context registry")
            if isinstance(model, FastgresModel) and model.n_features != 4 * len(context.schema.all_columns()):
                raise ValueError("Model input width does not match its prepared feature layout")

    def store(self, archive: Path | str) -> Path:
        """Save a self-contained bundle to a new or empty directory; return its catalog.

        The bundle includes encodings, contexts, feature ordering, hint ordering and
        defaults, training configuration, and models. It contains no connection
        credentials, query workload, or required reference to a training-label CSV.
        """
        self._validate_models()
        directory = Path(archive)
        schema = self.context_manager.db_schema
        assert isinstance(schema, DatabaseSchema)
        catalog = {
            "format_version": self._FORMAT_VERSION,
            "feature_format": self._FEATURE_FORMAT,
            "granularity": self.context_manager.ctx_granularity.name,
            "schema": schema.to_dict(),
            "contexts": [{"type": ctx.type, "schema": ctx.schema.to_dict(),
                          "features": list(ctx.schema)} for ctx in self.context_manager.ctx2q],
            "hints": [{"name": h.name, "db_instr": h.db_instr, "db_instr_val": h.db_instr_val}
                      for h in self._hint_library.hint_list],
            "model_params": self._model_params,
            "training_metrics": self._training_metrics,
        }
        serialized = json.dumps(catalog, indent=2)
        prepare_dir(directory)
        self.featurizer.statistics.store(directory / "encodings")
        self.model.save(directory / "models")
        # Write the catalog last, so incomplete saves cannot masquerade as a bundle.
        catalog_path = directory / "catalog.json"
        catalog_path.write_text(serialized, encoding="utf-8")
        return catalog_path

    @classmethod
    def pre_trained(cls, archive: Path | str, *,
                    database: pb.postgres.PostgresDatabase | None = None) -> FastgresOptimizer:
        """Load a bundle without the original workload, training labels, or database."""
        catalog_path = Path(archive)
        if catalog_path.is_dir():
            catalog_path /= "catalog.json"
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        if data.get("format_version") != cls._FORMAT_VERSION or data.get("feature_format") != cls._FEATURE_FORMAT:
            raise ValueError("Unsupported FASTgres bundle or feature format")
        schema = DatabaseSchema(data["schema"])
        contexts = []
        for item in data["contexts"]:
            context = ContextFactory.from_type(item["type"], DatabaseSchema(item["schema"]))
            if item["features"] != [list(feature) for feature in context.schema]:
                raise ValueError("Stored feature ordering does not match the context schema")
            contexts.append(context)
        manager = ContextManager.from_contexts(CtxGranularity[data["granularity"]], contexts, schema)
        optimizer = cls(database=database, context_granularity=manager.ctx_granularity,
                        hint_library=HintLibrary([Hint(**h) for h in data["hints"]]),
                        model_params=data["model_params"])
        optimizer._featurizer = FastgresFeaturization(
            None, manager, enc_dir=catalog_path.parent / "encodings", db_schema=schema)
        optimizer._model = FastgresContextModel(optimizer.featurizer, None, manager, **optimizer._model_params)
        optimizer._model.load(catalog_path.parent / "models")
        optimizer._validate_models()
        optimizer._training_metrics = data.get("training_metrics", {})
        return optimizer

    @classmethod
    def load_or_build(
        cls, archive: Path | str, *, database: pb.postgres.PostgresDatabase | None = None,
        workload: pb.Workload | None = None, labels: LabelSource | None = None,
        encoding_workload: pb.Workload | None = None,
        encodings: EncodingInformation | Path | str | None = None,
        schema: DatabaseSchema | None = None,
        context_granularity: CtxGranularity = CtxGranularity.SUPERTABLE,
        hint_library: HintLibrary = CORE_HINT_LIBRARY, model_params: Mapping[str, Any] | None = None,
    ) -> FastgresOptimizer:
        """Load an existing bundle, or explicitly prepare/train/store from supplied inputs.

        No labeling is performed. Existing bundles restore their own context,
        encodings, and hint library; training arguments are used only when building.
        """
        archive = Path(archive)
        if archive.is_file() or (archive / "catalog.json").is_file():
            return cls.pre_trained(archive, database=database)
        if archive.exists() and any(archive.iterdir()):
            raise ValueError("Directory is not a complete FASTgres bundle; choose a new or empty directory")
        if workload is None or labels is None:
            raise ValueError("Building a new optimizer requires a training workload and labels")
        optimizer = cls(database=database, context_granularity=context_granularity,
                        hint_library=hint_library, model_params=model_params)
        optimizer.prepare(encoding_workload if encoding_workload is not None else workload,
                          encodings=encodings, schema=schema)
        optimizer.train(workload, labels=labels)
        optimizer.store(archive)
        return optimizer

    def save(self, archive: Path | str) -> Path:
        return self.store(archive)

    @classmethod
    def load(cls, archive: Path | str, *,
             database: pb.postgres.PostgresDatabase | None = None) -> FastgresOptimizer:
        return cls.pre_trained(archive, database=database)

    def describe(self) -> pb.util.jsondict:
        return {"name": "FASTgres", "interface": type(self).__name__,
                "context_granularity": self._granularity.name,
                "prepared": self._featurizer is not None,
                "trained": self._model is not None and bool(self._model.models),
                "fallback": "warn-and-postgresql-defaults",
                "hints": [h.db_instr for h in self._hint_library.hint_list]}
