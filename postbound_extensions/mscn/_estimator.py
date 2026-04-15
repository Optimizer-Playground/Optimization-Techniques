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
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import postbound as pb
import torch

from ..util import PandasDataset, load_training_samples, wrap_logger
from ._featurizer import MscnFeaturizer
from ._misc import expand_dims, normalize_labels, qerror_loss, unnormalize_labels
from ._model import SetConv


@dataclass
class MscnHyperParams:
    """Hyper parameters used for the model training.

    By default, we use the same hyper parameters as the original MSCN paper. These are as follows:

    - training epochs: 70
    - learning rate: 0.001
    - batch size: 16
    """

    epochs: int = 70
    learning_rate: float = 0.001
    batch_size: int = 16

    @staticmethod
    def default() -> MscnHyperParams:
        return MscnHyperParams()

    def __json__(self) -> pb.util.jsondict:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }


class MscnEstimator(pb.CardinalityEstimator):
    """MSCN is a supervised learned cardinality estimator based on deep learning.

    An MSCN model must be trained on a large corpus of (query, cardinality) pairs before it can be
    used for estimation. Supply the samples using the `train` method.

    A key component of the estimator is an appropriate featurization method for the queries.
    The entire logic of this process is encapsulated in an `MscnFeaturizer`. The featurization
    scheme can be inferred from multiple different data sources. See the documentation of
    `MscnFeaturizer` for more details.

    Creating a new estimator
    ------------------------
    A fresh estimator can be obtained by simply creating a new instance of this class. By default,
    this will infer the featurization scheme from the database schema (see `MscnFeaturizer.online`).
    This can be customized by providing a pre-built featurizer.

    The `load_or_build` method provides a convenient entry point to avoid continuously re-training
    a model: It tries to load the estimator from a specific file path. If it does not exist, it will
    be created, trained, and stored at the desired location. Any subsequent calls to `load_or_build`
    will then load the pre-trained model.

    Parameters
    ----------
    model: Optional[SetConv]
        The actual MSCN model to use. If not provided, a new MSCN instance with random weights
        will be created.
    featurizer: Optional[MscnFeaturizer]
        The featurizer to use for encoding the queries. If not provided, it will be inferred from
        the database schema.
    database: Optional[pb.Database]
        The target database to run the optimized queries on. This is only used for the
        online-inference of the featurization scheme. If not provided, it will be loaded from the
        database pool. It is an error to use the MSCN estimator without an active database
        connection.
    verbose: bool | pb.util.Logger
        Whether to print verbose logs during the training and estimation process.

    Attributes
    ----------
    model: SetConv
        The underlying MSCN model used for estimation.
    featurizer: MscnFeaturizer
        The featurizer used for encoding the queries.

    See Also
    --------
    MscnFeaturizer :
        Our (adapted) query featurization scheme. It is based on the original MSCN paper but
        extended to handle more complex database schemas.

    References
    ----------
    .. Andreas Kipf et al.: "Learned Cardinalities: Estimating Correlated Joins with Deep Learning"
       (CIDR 2019) https://vldb.org/cidrdb/papers/2019/p101-kipf-cidr19.pdf
    """

    @staticmethod
    def pre_trained(
        catalog_path: Path | str,
        *,
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnEstimator:
        """Loads an MSCN estimator from disk.

        This is the inverse operation to `store`. Since `store` also takes care of persisting the
        featurizer, it is loaded as part of this process. The `catalog_path` must point to a valid
        JSON file containing the model info.

        MSCN always requires an active database connection, even if this is only used for the
        featurizer. Therefore, you either need to supply the database explicitly or ensure that
        there is a default database in the database pool.

        See Also
        ---------
        store : The inverse operation to this method. Persists an MSCN estimator to disk.
        """
        logger = wrap_logger(verbose)

        logger("Loading pre-trained MSCN estimator from", catalog_path)
        with open(catalog_path, "r") as f:
            catalog = json.load(f)

        database = database or pb.db.current_database()
        featurizer = MscnFeaturizer.pre_built(catalog_path, verbose=verbose)

        logger("Loading MSCN model from", catalog["mscn_model"])
        weights = torch.load(catalog["mscn_model"])
        model = SetConv(
            n_tables=featurizer.n_tables,
            n_columns=featurizer.n_columns,
            n_operators=featurizer.n_operators,
            n_joins=featurizer.n_joins,
            verbose=verbose,
        )
        model.load_state_dict(weights)

        raw_hyper_params = catalog.get("hyper_parameters")
        training_params = (
            MscnHyperParams(**raw_hyper_params)
            if raw_hyper_params is not None
            else None
        )
        training_metrics = catalog.get("training_metrics", {})

        estimator = MscnEstimator(
            model=model, featurizer=featurizer, database=database, verbose=verbose
        )
        estimator._training_params = training_params
        estimator._training_metrics = training_metrics

        return estimator

    @staticmethod
    def load_or_build(
        catalog_path: Path | str,
        *,
        samples: pd.DataFrame | Path | str,
        workload: Optional[pb.Workload] = None,
        database: Optional[pb.Database] = None,
        training_params: MscnHyperParams = MscnHyperParams.default(),
        verbose: bool | pb.util.Logger = False,
    ) -> MscnEstimator:
        """Integrated model training and storage procedure.

        If an MSCN estimator has already been stored to `catalog_path`, it will simply be loaded.
        Otherwise, a new model will be created and trained using the `samples` and according to the
        `training_params`. The featurizer is inferred based on the samples and an optional
        evaluation workload.

        The `workload` is only used to fine-tune the featurizer. If it is not provided, the
        featurization scheme will only be based on the training samples.

        Once the model has been trained, it will be stored at `catalog_path` along with the
        featurizer.

        See Also
        ---------
        MscnFeaturizer.infer_from_samples : Inference logic for the featurizer
        """
        logger = wrap_logger(verbose)
        catalog_path = Path(catalog_path)
        if catalog_path.exists():
            logger("Catalog exists, loading pre-trained MSCN model")
            return MscnEstimator.pre_trained(catalog_path, verbose=verbose)

        logger("Catalog not found, training new MSCN model")
        database = database or pb.db.current_database()
        samples = load_training_samples(samples, query_col="query", verbose=verbose)

        # See comment in MscnFeaturizer.infer_from_samples() on why we have to use this
        # strategy instead of calling infer_from_workload() or load_or_build()
        logger("Determining MSCN features")
        featurizer = MscnFeaturizer.infer_from_samples(
            samples, workload=workload, database=database, verbose=verbose
        )

        estimator = MscnEstimator(
            featurizer=featurizer, database=database, verbose=verbose
        )
        logger("Training new MSCN estimator")
        estimator.train(samples, hyper_params=training_params)
        estimator.store(catalog_path)
        return estimator

    def __init__(
        self,
        *,
        model: Optional[SetConv] = None,
        featurizer: Optional[MscnFeaturizer] = None,
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> None:
        super().__init__()
        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._cuda else "cpu")
        self._log = wrap_logger(verbose)

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
        if self._cuda:
            self._log("Using CUDA for MSCN model")
            self.model = self.model.cuda()

        self._training_params: MscnHyperParams | None = None
        self._training_metrics: dict = {"status": "untrained"}
        self._verbose = verbose

    @property
    def training_metrics(self) -> pb.train.TrainingMetrics:
        """Get the training metrics of the last training run."""
        return self._training_metrics

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

        featurized = self.featurizer.encode_single(query_fragment).to(self._device)

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

    def fit_samples(self, samples: pb.train.TrainingData) -> pb.train.TrainingMetrics:
        return self.train(samples.as_df())

    def sample_spec(self) -> pb.train.TrainingSpec:
        return pb.train.TrainingSpec(["query", "cardinality"])

    def sample_fit_completed(self) -> bool:
        return self._training_metrics.get("status") == "completed"

    def train(
        self,
        samples: pd.DataFrame,
        *,
        query_col: str = "query",
        label_col: str = "cardinality",
        hyper_params: MscnHyperParams = MscnHyperParams.default(),
    ) -> pb.train.TrainingMetrics:
        """Trains the MSCN model on a specific set of training samples.

        If the model has already been trained, the additional samples will be used as a fine-tuning
        step.

        Parameters
        ----------
        samples: pd.DataFrame
            The training samples to use for training the model. This must be a DataFrame containing
            one column with the SQL queries and another column containing the cardinalities of those
            queries. Additional columns are ignored. If the query column contains parsed PostBOUND
            SQL queries, these will be used directly. Otherwise, queries will be parsed.
        query_col: str
            The name of the column containing the SQL queries. Default is "query".
        label_col: str
            The name of the column containing the cardinality labels. Default is "cardinality".
        hyper_params: MscnHyperParams
            The hyper parameters to use for training the model. By default, we use the same
            hyper parameters as the original MSCN paper.

        See Also
        --------
        MscnHyperParams
        """
        if not len(samples):
            metrics: dict = {"status": "untrained"}
            self._training_metrics = metrics
            return metrics

        self._log("Preparing training dataset")
        metrics: dict = {
            "n_samples": len(samples),
            "hyper_params": hyper_params,
            "loss": [],
        }

        min_card = self.featurizer.norm_min_card
        max_card = self.featurizer.norm_max_card
        if isinstance(samples[query_col].iloc[0], pb.SqlQuery):
            queries = samples[query_col]
        else:
            self._log("Parsing queries")
            queries = samples[query_col].map(pb.parse_query)

        self._log("Featurizing queries")
        featurized = self.featurizer.encode_batch(queries)
        training_df = pd.DataFrame(featurized)
        self._log("Normalizing labels")
        training_df["label"] = normalize_labels(samples[label_col], min_card, max_card)

        training_data: torch.utils.data.Dataset = PandasDataset(training_df)

        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=hyper_params.learning_rate
        )

        data_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=hyper_params.batch_size,
            pin_memory=True,
        )

        self._log("Starting training")
        start_time = time.perf_counter_ns()

        for epoch in range(hyper_params.epochs):
            loss_total = 0.0
            for batch in data_loader:
                features = self._move_features(batch[:-1])
                labels = batch[-1].to(self._device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(*features)
                loss = qerror_loss(outputs, labels.float(), min_card, max_card)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()

            epoch_str = str(epoch + 1).rjust(len(str(hyper_params.epochs)))
            self._log(
                f"Epoch: {epoch_str} / {hyper_params.epochs} :: loss = {loss_total}"
            )
            metrics["loss"].append(loss_total)

        end_time = time.perf_counter_ns()
        elapsed_time = (end_time - start_time) / 1e9
        metrics["training_time"] = elapsed_time
        metrics["status"] = "completed"

        self._training_params = hyper_params
        self._training_metrics = metrics
        self.model.eval()

        return metrics

    def store(
        self,
        catalog: Path | str,
        *,
        encoder_dir: Optional[Path | str] = None,
        model_file: Optional[Path | str] = None,
    ) -> Path:
        """Persists model weights and featurization info at the specified location.

        Parameters
        ----------
        catalog: Path | str
            The path to the JSON file where the model info should be stored. If the file already
            exists, it will be overwritten. If this is a directory, it will be augmented to
            "mscn-catalog-{schema}.json", where {schema} is the name of the database schema.
            The catalog will contain both model and featurization info.
        encoder_dir: Optional[Path | str]
            Directory where the featurizer should store its raw column encoders. By default, this
            is the same directory as the one containing the catalog.
        model_file: Optional[Path | str]
            The path where the model weights should be stored. This defaults to "mscn-{schema}.pt"
            and will be created in the same directory as the catalog.

        See Also
        --------
        pre_trained : The inverse operation to this method. Loads an MSCN estimator from disk.
        MscnFeaturizer.store : Export logic for the featurizer
        """
        schema = self._database.database_name()
        catalog = Path(catalog)
        if catalog.is_dir():
            catalog = catalog / f"mscn-catalog-{schema}.json"

        if encoder_dir is None:
            encoder_dir = catalog.parent
        else:
            encoder_dir = Path(encoder_dir)

        if model_file is None:
            model_file = catalog.parent / f"mscn-{schema}.pt"
        else:
            model_file = Path(model_file)

        catalog.parent.mkdir(parents=True, exist_ok=True)
        model_file.parent.mkdir(parents=True, exist_ok=True)

        self.featurizer.store(catalog, encoder_dir=encoder_dir)

        self._log("Storing MSCN model to", model_file)
        weights = self.model.state_dict()
        torch.save(weights, model_file)

        self._log("Finalizing catalog at", catalog)
        with open(catalog, "r+") as f:
            catalog = json.load(f)
            catalog["mscn_model"] = str(model_file)
            catalog["hyper_parameters"] = (
                self._training_params.__json__() if self._training_params else None
            )
            catalog["training_metrics"] = self._training_metrics
            f.seek(0)
            pb.util.to_json_dump(catalog, f)

        return catalog

    def describe(self) -> pb.util.jsondict:
        return {"name": "MSCN", "hyperparameters": self._training_params}

    def _move_features(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        moved: list[torch.Tensor] = []
        for feature in features:
            feature = feature.to(self._device, non_blocking=True)
            feature.requires_grad_(True)
            moved.append(feature)
        return moved
