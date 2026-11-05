
"""FASTgres with FASTlabel: adapted optimizer hinting for PostBOUND.

FastgresOptimizer is the public workflow for preparation, explicit labeling,
training, prediction, and portable model bundles. It defaults to one supertable
context and warns before returning PostgreSQL defaults for uncovered contexts.

The component classes remain available for experiments with different context
policies, encodings, and model configurations. FastgresEstimator and
FastgresSetting retain the earlier interface. See README.md for an introduction,
usage examples, and references to the original FASTgres and FASTlabel papers.
"""


from ._fastgres import FastgresEstimator, FastgresSetting
from ._optimizer import FastgresContextWarning, FastgresOptimizer
from ._util import FgPbConverter
from .context import (
    ColumnContext,
    ContextManager,
    CtxGranularity,
    DatabaseSchema,
    SchemaContext,
    SuperTableContext,
    TableContext,
)
from .featurization import (
    DatabaseConnection,
    EncodingInformation,
    FastgresFeaturization,
)
from .hinting import CORE_HINT_LIBRARY, HintSet, HintSetFactory
from .labeling import (
    FastgresLabelProvider,
    FastLabelSettings,
    QueryLabeling,
    WorkloadLabeling,
    WorkloadLabelSettings,
)
from .model import FastgresContextModel, FastgresModel

__all__ = [
    "CORE_HINT_LIBRARY",
    "ColumnContext",
    "ContextManager",
    "CtxGranularity",
    "DatabaseConnection",
    "DatabaseSchema",
    "EncodingInformation",
    "FastLabelSettings",
    "FastgresContextModel",
    "FastgresContextWarning",
    "FastgresEstimator",
    "FastgresFeaturization",
    "FastgresLabelProvider",
    "FastgresModel",
    "FastgresOptimizer",
    "FastgresSetting",
    "FgPbConverter",
    "HintSet",
    "HintSetFactory",
    "QueryLabeling",
    "SchemaContext",
    "SuperTableContext",
    "TableContext",
    "WorkloadLabelSettings",
    "WorkloadLabeling",
]
