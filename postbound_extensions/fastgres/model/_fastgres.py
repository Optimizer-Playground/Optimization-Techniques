from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import onnx
import onnxruntime as ort
import postbound as pb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier

from .._util import load_json, prepare_dir, save_json
from ..context import Context, ContextFactory, ContextManager, DatabaseSchema
from ..featurization import FastgresFeaturization
from ..labeling import FastgresLabelProvider


class FastgresModel:

    def __init__(self, **gb_kwargs):
        self._live_model: GradientBoostingClassifier = GradientBoostingClassifier(**gb_kwargs)
        self._onnx_model: onnx.ModelProto | None = None

    def fit(self, x, y):
        if self._onnx_model is not None:
            raise RuntimeError("Cannot fit loaded onnx model.")
        self._live_model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if self._onnx_model is None:
            return self._live_model.predict(x)

        sess = ort.InferenceSession(self._onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        return np.asarray(sess.run(None, {input_name: x.astype(np.float32)})[0])

    def save(self, path: Path):
        if self._onnx_model is not None:
            onnx_model = self._onnx_model
        elif not hasattr(self._live_model, "n_features_in_"):
            raise RuntimeError("Model must be fitted before saving to ONNX.")
        else:
            initial_type = [('float_input', FloatTensorType([None, self._live_model.n_features_in_]))]
            onnx_model = convert_sklearn(self._live_model, initial_types=initial_type)
        with path.open("wb") as f:
            f.write(onnx_model.SerializeToString())

    def load(self, path: Path):
        with path.open("rb") as f:
            self._onnx_model = onnx.load_model(f)

    @property
    def n_features(self) -> int:
        if self._onnx_model is not None:
            return self._onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        count = getattr(self._live_model, "n_features_in_", None)
        if count is None:
            raise ValueError("Model must be fitted before its feature width is known")
        return int(count)


class FastgresContextModel:
    @dataclass
    class IntegerModel:
        _model: int | None = field(default=None, init=False)

        def fit(self, x, y) -> None:
            labels = np.unique(y)
            if len(labels) != 1:
                raise ValueError(f"Integer model requires exactly one distinct label: {y}.")
            self._model = int(labels[0])

        def predict(self, x) -> list[int]:
            if self._model is None:
                raise ValueError("Integer model must be fitted or loaded before prediction")
            return [self._model for _ in x]

        def load(self, path: Path):
            model_dict = load_json(path)
            self.fit(None, np.asarray(model_dict["model"]).reshape(-1))

        def save(self, path: Path):
            if self._model is None:
                raise ValueError("Integer model must be fitted or loaded before saving")
            save_json({"model": self._model}, path)

    def __init__(
            self,
            featurizer: FastgresFeaturization,
            label_provider: FastgresLabelProvider | None,
            context_manager: ContextManager,
            **gb_kwargs,):
        super().__init__()
        self.cm = context_manager
        self.gb_kwargs = gb_kwargs
        self.featurizer = featurizer
        self.label_provider = label_provider
        self._models: dict[Context, FastgresModel | FastgresContextModel.IntegerModel] = {}

    @property
    def models(self) -> Mapping[Context, FastgresModel | FastgresContextModel.IntegerModel]:
        return MappingProxyType(self._models)


    def create_model(self, context: Context) -> FastgresModel:
        model = FastgresModel(**self.gb_kwargs)
        self._models[context] = model
        return model

    def fit(self, workload: pb.Workload):
        if not workload:
            raise ValueError("Cannot fit FastGRES on an empty workload")
        if self.label_provider is None:
            raise ValueError("Training requires labels; loaded models can predict without them")
        models: dict[Context, FastgresModel | FastgresContextModel.IntegerModel] = {}
        wl_queries = workload.queries()
        ctx2q, q2ctx = self.cm.classify_queries(wl_queries)
        for context, query_set in ctx2q.items():
            sorted_labels = sorted([workload.label_of(q) for q in query_set])
            sorted_queries = [workload[qn] for qn in sorted_labels]
            ctx_train_x = self.featurizer.transform(sorted_queries)
            ctx_train_y = self.label_provider.get_labels(sorted_labels)
            model = self.IntegerModel() if len(np.unique(ctx_train_y)) == 1 else FastgresModel(**self.gb_kwargs)
            model.fit(ctx_train_x, ctx_train_y)
            models[context] = model
        self._models = models

    def predict(self, query: pb.SqlQuery) -> int:
        context = self.cm[query]
        if context is None:
            raise ValueError(f"No context registered for query: {query}")
        model = self._models.get(context, None)
        if model is None:
            raise ValueError(f"No model found for context: {context}")
        feature = self.featurizer.transform_single(query)
        return int(model.predict(feature.reshape(1, -1))[0])

    def predict_queries(
            self,
            queries: Iterable[pb.SqlQuery],
    ) -> list[int]:
        return [self.predict(q) for q in queries]

    def save(self, path: Path):
        if not self._models:
            raise ValueError("Model must be fitted or loaded before saving")
        prepare_dir(path)
        manifest = dict()
        for context, model in self._models.items():
            ctx_hash = str(hash(context))
            context_path = path / ctx_hash
            # context_path.mkdir(parents=True, exist_ok=True)
            prepare_dir(context_path)
            if isinstance(model, self.IntegerModel):
                model.save(context_path / "model.json")
                manifest[ctx_hash] = {"schema": context.schema.to_dict(), "ctx_type": context.type, "type": "json"}
            else:
                model.save(context_path / "model.onnx")
                manifest[ctx_hash] = {"schema": context.schema.to_dict(), "ctx_type": context.type, "type": "onnx"}
        with (path / "manifest.json").open("w") as f:
            json.dump(manifest, f)

    def load(self, path: Path):
        with (path / "manifest.json").open("r") as f:
            manifest = json.load(f)
        models: dict[Context, FastgresModel | FastgresContextModel.IntegerModel] = {}
        for ctx_hash, ctx_dict in manifest.items():
            ctx_schema: DatabaseSchema = DatabaseSchema(ctx_dict["schema"])
            ctx_type: str = ctx_dict["ctx_type"]
            model_type: str = ctx_dict["type"]
            context = ContextFactory.from_type(ctx_type, ctx_schema)
            if model_type == "json":
                model = self.IntegerModel()
                model.load(path / ctx_hash / "model.json")
            elif model_type == "onnx":
                model = FastgresModel(**self.gb_kwargs)
                model.load(path / ctx_hash / "model.onnx")
            else:
                raise ValueError(f"Unknown saved FastGRES model type: {model_type}")
            models[context] = model
        self._models = models
