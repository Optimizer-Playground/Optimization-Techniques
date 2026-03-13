import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import postbound as pb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier

from .._util import load_json, save_json, prepare_dir
from ..context import Context, ContextFactory
from ..context import ContextManager
from ..context import DatabaseSchema
from ..featurization import FastgresFeaturization
# from ._base import BaseModel
# from ._context_base import ContextModel
from ..labeling import FastgresLabelProvider


class FastgresModel:

    def __init__(self, **gb_kwargs):
        self._live_model: GradientBoostingClassifier = GradientBoostingClassifier(**gb_kwargs)
        self._onnx_model: Optional[onnx.ModelProto] = None

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
        return sess.run(None, {input_name: x.astype(np.float32)})[0]

    def save(self, path: Path):
        if not hasattr(self._live_model, "n_features_in_"):
            raise RuntimeError("Model must be fitted before saving to ONNX.")
        initial_type = [('float_input', FloatTensorType([None, self._live_model.n_features_in_]))]
        onnx_model = convert_sklearn(self._live_model, initial_types=initial_type)
        with path.open("wb") as f:
            f.write(onnx_model.SerializeToString())

    def load(self, path: Path):
        with path.open("rb") as f:
            self._onnx_model = onnx.load_model(f)


class FastgresContextModel:
    @dataclass
    class IntegerModel:
        _model: int = field(init=False)

        def fit(self, x, y) -> None:
            if len(set(y)) > 1:
                raise ValueError(f"Cannot fit integer model to variate labels: {y}.")
            self._model = y

        def predict(self, x) -> Any:
            return [self._model for _ in x]

        def load(self, path: Path):
            model_dict = load_json(path)
            self._model = model_dict["model"]

        def save(self, path: Path):
            save_json({"model": self._model}, path)

    def __init__(
            self,
            featurizer: FastgresFeaturization,
            label_provider: FastgresLabelProvider,
            context_manager: ContextManager,
            **gb_kwargs,):
        super().__init__()
        self.cm = context_manager
        self.gb_kwargs = gb_kwargs
        self.featurizer = featurizer
        self.label_provider = label_provider
        self._models: dict[Context, FastgresModel | FastgresContextModel.IntegerModel] = {}


    def create_model(self, context: Context) -> FastgresModel:
        self._models[context] = FastgresModel(**self.gb_kwargs)
        return self._models[context]

    def fit(self, workload: pb.Workload):
        wl_queries = workload.queries()
        ctx2q, q2ctx = self.cm.classify_queries(wl_queries)
        for context, query_set in ctx2q.items():
            sorted_labels = sorted([workload.label_of(q) for q in query_set])
            sorted_queries = [workload[qn] for qn in sorted_labels]
            ctx_train_x = self.featurizer.transform(sorted_queries)
            ctx_train_y = self.label_provider.get_labels(sorted_labels)
            if len(ctx_train_y) < 2:
                self._models[context] = self.IntegerModel()
            model = self.create_model(context)
            self._models[context] = model
            model.fit(ctx_train_x, ctx_train_y)

    def predict(self, query: pb.SqlQuery) -> Any:
        context = self.cm[query]
        model = self._models.get(context, None)
        if model is None:
            raise ValueError(f"No model found for context: {context}")
        feature = self.featurizer.transform_single(query)
        return model.predict(feature.reshape(1, -1))[0]

    def predict_queries(
            self,
            queries: Iterable[pb.SqlQuery],
    ) -> Any:
        return [self.predict(q) for q in queries]

    def save(self, path: Path):
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
        for ctx_hash, ctx_dict in manifest.items():
            ctx_schema: DatabaseSchema = DatabaseSchema(ctx_dict["schema"])
            ctx_type: str = ctx_dict["ctx_type"]
            model_type: str = ctx_dict["type"]
            context = ContextFactory.from_type(ctx_type, ctx_schema)
            if model_type == "json":
                model = self.IntegerModel()
                model.load(path / ctx_hash / "model.json")
            else:
                model = self.create_model(context)
                model.load(path / ctx_hash / "model.onnx")
            self._models[context] = model
