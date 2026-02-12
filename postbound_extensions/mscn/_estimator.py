"""
MSCN PostBOUND adapter
    Created as part of the Optimization Techniques project.

The MIT License

Copyright (c) 2026 Rico Bergmann

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import postbound as pb
import torch

from ..util import PandasDataset, wrap_logger
from ._featurizer import MscnFeaturizer
from ._model import SetConv
from ._util import expand_dims, normalize_labels, qerror_loss, unnormalize_labels


class MscnEstimator(pb.CardinalityEstimator):
    @staticmethod
    def pre_trained(
        catalog_path: Path | str,
        *,
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnEstimator:
        with open(catalog_path, "r") as f:
            catalog = json.load(f)

        database = database or pb.db.current_database()
        featurizer = MscnFeaturizer.pre_built(catalog_path, verbose=verbose)

        model_file = Path(catalog["mscn_model"])
        model_program = torch.export.load(model_file)
        model = model_program.module()

        estimator = MscnEstimator(
            model=model, featurizer=featurizer, database=database, verbose=verbose
        )

        estimator._sample_query = pb.parse_query(catalog["export_query"])
        return estimator

    @staticmethod
    def load_or_build(
        catalog_path: Path | str,
        *,
        samples: pd.DataFrame,
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnEstimator:
        catalog_path = Path(catalog_path)
        if catalog_path.exists():
            return MscnEstimator.pre_trained(catalog_path, verbose=verbose)

        database = database or pb.db.current_database()
        featurizer = MscnFeaturizer.load_or_build(
            catalog_path, database=database, verbose=verbose
        )
        estimator = MscnEstimator(
            featurizer=featurizer, database=database, verbose=verbose
        )
        estimator.train(samples)
        estimator.store(catalog_path)
        return estimator

    def __init__(
        self,
        *,
        model: Optional[SetConv | torch.fx.GraphModule] = None,
        featurizer: Optional[MscnFeaturizer] = None,
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> None:
        super().__init__()
        self._database = database or pb.db.current_database()
        if featurizer is None:
            featurizer = MscnFeaturizer.online(self._database, verbose=verbose)

        self.featurizer = featurizer
        self.model = model or SetConv(
            n_tables=self.featurizer.n_tables,
            n_columns=self.featurizer.n_columns,
            n_operators=self.featurizer.n_operators,
            n_joins=self.featurizer.n_joins,
            verbose=verbose,
        )

        self._sample_query: pb.SqlQuery | None = None
        self._log = wrap_logger(verbose)
        self._verbose = verbose

    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        intermediate: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        intermediate = pb.util.enlist(intermediate)
        query_fragment = pb.transform.extract_query_fragment(query, intermediate)
        if not query_fragment:
            raise ValueError(f"Query fragment not found for query {query}")
        query_fragment = pb.transform.as_star_query(query_fragment)

        featurized = self.featurizer.encode_single(query_fragment)

        with torch.no_grad():
            output = self.model(
                expand_dims(featurized.tables),
                expand_dims(featurized.joins),
                expand_dims(featurized.predicates),
                expand_dims(featurized.tables_mask),
                expand_dims(featurized.joins_mask),
                expand_dims(featurized.predicates_mask),
            )

        min_card = self.featurizer.norm_min_card
        max_card = self.featurizer.norm_max_card
        normalized_output = unnormalize_labels(output, min_card, max_card)
        estimated_value = normalized_output.item()
        return pb.Cardinality(estimated_value)

    def train(
        self,
        samples: pd.DataFrame,
        *,
        query_col: str = "query",
        label_col: str = "cardinality",
        epochs: int = 70,
        lr: float = 0.001,
        batch_size: int = 16,
    ) -> None:
        if not len(samples):
            return

        self._log("Preparing training dataset")
        min_card = self.featurizer.norm_min_card
        max_card = self.featurizer.norm_max_card
        queries = samples[query_col].map(pb.parse_query)
        featurized = self.featurizer.encode_batch(queries)
        featurized_components = [asdict(featurized) for featurized in featurized]
        training_df = pb.util.df.as_df(featurized_components)
        training_df["label"] = normalize_labels(samples[label_col], min_card, max_card)

        training_data: torch.utils.data.Dataset = PandasDataset(training_df)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)

        self._log("Starting training")
        for epoch in range(epochs):
            loss_total = 0.0
            for batch in data_loader:
                features = batch[:-1]
                labels = batch[-1]
                optimizer.zero_grad()
                outputs = self.model(*features)
                loss = qerror_loss(outputs, labels.float(), min_card, max_card)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()

            epoch_str = str(epoch + 1).rjust(len(str(epochs)))
            self._log(f"Epoch: {epoch_str} / {epochs} :: loss = {loss_total}")

        self.model.eval()
        self._sample_query = queries.iloc[0] if not queries.empty else None

    def store(
        self,
        catalog: Path | str,
        *,
        encoder_dir: Optional[Path | str] = None,
        model_file: Optional[Path | str] = None,
    ) -> Path:
        if self._sample_query is None:
            raise RuntimeError("Estimator must be trained before it can be stored")

        catalog = Path(catalog)
        if encoder_dir is None:
            encoder_dir = catalog.parent
        else:
            encoder_dir = Path(encoder_dir)
        if model_file is None:
            schema = self._database.database_name()
            model_file = catalog.parent / f"mscn-{schema}.pt2"
        else:
            model_file = Path(model_file)

        catalog.parent.mkdir(parents=True, exist_ok=True)
        model_file.parent.mkdir(parents=True, exist_ok=True)

        self.featurizer.store(catalog, encoder_dir=encoder_dir)

        self._log("Creating exportable program for MSCN model")
        sample_input = self.featurizer.encode_single(self._sample_query)

        program = torch.export.export(
            self.model,
            (
                expand_dims(sample_input.tables),
                expand_dims(sample_input.joins),
                expand_dims(sample_input.predicates),
                expand_dims(sample_input.tables_mask),
                expand_dims(sample_input.joins_mask),
                expand_dims(sample_input.predicates_mask),
            ),
        )

        self._log("Storing MSCN model to", model_file)
        torch.export.save(program, model_file)

        self._log("Finalizing catalog at", catalog)
        with open(catalog, "r+") as f:
            catalog = json.load(f)
            catalog["mscn_model"] = str(model_file)
            catalog["export_query"] = str(self._sample_query)
            f.seek(0)
            json.dump(catalog, f)

        return model_file

    def describe(self) -> pb.util.jsondict:
        return {"name": "MSCN"}
