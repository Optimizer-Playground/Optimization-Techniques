from __future__ import annotations

from pathlib import Path

import postbound as pb
from postbound import JoinTree, PhysicalOperatorAssignment, QueryPlan, SqlQuery

from ._util import FgPbConverter
from .context import ContextManager, CtxGranularity, DatabaseSchema
from .featurization import DatabaseConnection, EncodingInformation, FastgresFeaturization
from .hinting import HintLibrary, HintSetFactory
from .labeling import FastgresLabelProvider
from .model import FastgresContextModel


class FastgresSetting:

    def __init__(
            self,
            pb_connection: pb.postgres.PostgresDatabase,
            encoding_information_path: Path | str,
            training_data_path: Path | str,
            context_granularity: CtxGranularity,
            hint_library: HintLibrary,
            encoding_workload: pb.Workload,
            feature_skip_columns: dict[str, list[str]] | None = None,
    ):
        self.dbc = pb_connection
        self.enc_inf_path = Path(encoding_information_path)
        self.label_path = Path(training_data_path)
        self.ctx = context_granularity
        self.hint_lib = hint_library
        self.enc_wl = encoding_workload
        self.skip_dict = feature_skip_columns


class FastgresEstimator(pb.OperatorSelection, pb.CompleteOptimizationAlgorithm):
    """Predict global optimizer hints using a contextual FASTgres model.

    Initialize the featurizer, then build and fit a model or load a saved model before
    selecting operators. Complete optimization delegates the hinted query to PostgreSQL.
    """

    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        operators = self.select_physical_operators(query)
        hinted_query = self._dbc.hinting().generate_hints(query, physical_operators=operators)
        return self._dbc.optimizer().query_plan(hinted_query)

    def select_physical_operators(self, query: SqlQuery,
                                  join_order: JoinTree | None = None) -> PhysicalOperatorAssignment:
        prediction = self._hsf.hint_set(self._require_model().predict(query))
        return self._conv.fg_to_pb_hint_set(prediction)

    def __init__(self, setting: FastgresSetting) -> None:
        super().__init__()
        self.setting = setting
        self._dbc = self.setting.dbc
        self._conv = FgPbConverter()
        self._label_provider = FastgresLabelProvider(self.setting.label_path)
        self._ctx_manager = ContextManager(self.setting.ctx, self.setting.enc_wl.queries(),
                                           DatabaseSchema.from_dbc(self._dbc))
        self._enc_skipped = dict() if self.setting.skip_dict is None else self.setting.skip_dict
        self._hsf = HintSetFactory(self.setting.hint_lib)

        self._featurizer: FastgresFeaturization | None = None
        self._model: FastgresContextModel | None = None

    def load_or_build_featurizer(self, enc_dir: Path | str | None = None) -> None:
        enc_dir = self.setting.enc_inf_path if enc_dir is None else Path(enc_dir)
        if enc_dir.exists() and not enc_dir.is_dir():
            raise NotADirectoryError(enc_dir)
        if enc_dir.exists() and any(enc_dir.iterdir()):
            self._featurizer = FastgresFeaturization(self._dbc, self._ctx_manager, enc_dir=enc_dir)
        else:
            enc_info = EncodingInformation.build(DatabaseConnection(self._dbc), self.setting.enc_wl, self._enc_skipped)
            self._featurizer = FastgresFeaturization(self._dbc, self._ctx_manager, enc_info=enc_info)

    def load_or_build_model(self, model_dir: Path | str | None = None) -> None:
        if self._featurizer is None:
            raise ValueError("Initialize the featurizer before building or loading a model")
        fg_model = FastgresContextModel(
            featurizer=self._featurizer,
            label_provider=self._label_provider,
            context_manager=self._ctx_manager,
            max_depth=1000,
            n_estimators=100,
        )
        if model_dir is not None:
            fg_model.load(Path(model_dir))
        self._model = fg_model

    def _require_model(self) -> FastgresContextModel:
        if self._model is None:
            raise ValueError("Build or load a FastGRES model first")
        return self._model

    def fit(self, training_workload: pb.Workload) -> None:
        self._require_model().fit(training_workload)

    def predict(self, test_workload: pb.Workload) -> list[int]:
        return self._require_model().predict_queries(test_workload.queries())

    def save(self, path: Path) -> None:
        self._require_model().save(path)

    def load(self, path: Path) -> None:
        self.load_or_build_model(path)

    def describe(self) -> pb.util.jsondict:
        return {
            "name": "Fastgres",
            "version": "1.0",
            "database": self._dbc.describe(),
            "hints": [hint.db_instr for hint in self._hsf.hint_library.hint_list]
        }
