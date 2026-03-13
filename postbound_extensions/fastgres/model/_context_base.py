# import json
# from abc import ABC, abstractmethod
# from pathlib import Path
# from typing import Dict, Any, Collection
#
# from ..context import Context, ContextFactory, DatabaseSchema
# from ._base import BaseModel
#
#
# class ContextModel(ABC):
#     def __init__(self):
#         self._models: Dict[Context, BaseModel] = {}
#
#     def fit(self, x: Collection[Any], y: Collection[Any], context: Context) -> None:
#         model = self._models.get(context, None)
#         if model is None:
#             model = self.create_model(context)
#             self._models[context] = model
#         model.fit(x, y)
#
#     def predict(self, x: Collection[Any], context: Context) -> Any:
#         model = self._models.get(context, None)
#         if model is None:
#             raise ValueError(f"No model found for context: {context}")
#         return model.predict(x)
#
#     def save(self, path: Path):
#         manifest = dict()
#         for context, model in self._models.items():
#             ctx_hash = str(hash(context))
#             context_path = path / ctx_hash
#             context_path.mkdir(parents=True, exist_ok=True)
#             model.save(context_path / "model.onnx")
#             manifest[ctx_hash] = {"schema": context.schema.to_dict(), "ctx_type": context.type}
#         with (path / "manifest.json").open("w") as f:
#             json.dump(manifest, f)
#
#     def load(self, path: Path):
#         with (path / "manifest.json").open("r") as f:
#             manifest = json.load(f)
#         for ctx_hash, ctx_dict in manifest.items():
#             ctx_schema: DatabaseSchema = DatabaseSchema(ctx_dict["schema"])
#             ctx_type: str = ctx_dict["ctx_type"]
#             context = ContextFactory.from_type(ctx_type, ctx_schema)
#             model = self.create_model(context)
#             model.load(path / ctx_hash / "model.onnx")
#             self._models[context] = model
#
#     @abstractmethod
#     def create_model(self, context: Context) -> BaseModel:
#         """Instantiate a new BaseModel for the given context."""
#         raise NotImplementedError
