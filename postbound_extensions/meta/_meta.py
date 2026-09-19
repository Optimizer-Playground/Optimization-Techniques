from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import postbound as pb


def _load_target[T](mod_path: Path, target: type[T]) -> T | None:
    spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    matches = [member for _, member in inspect.getmembers(mod) if isinstance(member, target)]
    match matches:
        case []:
            return None
        case [match]:
            return match
        case _:
            return None


def load_cardinality_estimator(mod_path: str | Path) -> pb.CardinalityEstimator:
    """Loads a `CardinalityEstimator` instance from a Python module.

    The target module must contain exactly one cardinality estimator instance. The return value of this function
    is the member variable. It is not sufficient to just define a custom cardinality estimator subclass, the module
    must also handle instantiation of the estimator.

    Warnings
    --------
    In order to determine the cardinality estimator, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """
    mod_path = Path(mod_path)
    estimator = _load_target(mod_path, pb.CardinalityEstimator)
    if estimator is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one CardinalityEstimator instance.")
    return estimator


def load_cost_model(mod_path: str | Path) -> pb.CostModel:
    """Loads a `CostModel` instance from a Python module.

    The target module must contain exactly one cost model instance. The return value of this function is the member
    variable. It is not sufficient to just define a custom cost model subclass, the module must also handle
    instantiation of the model.

    Warnings
    --------
    In order to determine the cost model, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """
    mod_path = Path(mod_path)
    cost_model = _load_target(mod_path, pb.CostModel)
    if cost_model is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one CostModel instance.")
    return cost_model


def load_enumerator(mod_path: str | Path) -> pb.PlanEnumerator:
    """Loads a `PlanEnumerator` instance from a Python module.

    The target module must contain exactly one plan enumerator instance. The return value of this function is the
    member variable. It is not sufficient to just define a custom enumerator subclass, the module must also handle
    instantiation of the enumerator.

    Warnings
    --------
    In order to determine the enumerator, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """
    mod_path = Path(mod_path)
    enumerator = _load_target(mod_path, pb.PlanEnumerator)
    if enumerator is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one Enumerator instance.")
    return enumerator


def load_join_order_optimizer(mod_path: str | Path) -> pb.JoinOrderOptimization:
    """Loads a `JoinOrderOptimization` instance from a Python module.

    The target module must contain exactly one optimizer instance. The return value of this function is the
    member variable. It is not sufficient to just define a custom optimizer subclass, the module must also handle
    instantiation of the optimizer.

    Warnings
    --------
    In order to determine the optimizer, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """
    mod_path = Path(mod_path)
    optimizer = _load_target(mod_path, pb.JoinOrderOptimization)
    if optimizer is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one JoinOrderOptimization instance.")
    return optimizer


def load_operator_selection(mod_path: str | Path) -> pb.PhysicalOperatorSelection:
    """Loads a `PhysicalOperatorSelection` instance from a Python module.

    The target module must contain exactly one operator selection instance. The return value of this function is the
    member variable. It is not sufficient to just define a custom operator selection subclass, the module must also
    handle instantiation of the optimizer.

    Warnings
    --------
    In order to determine the operator selection, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """
    mod_path = Path(mod_path)
    operator_selection = _load_target(mod_path, pb.PhysicalOperatorSelection)
    if operator_selection is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one PhysicalOperatorSelection instance.")
    return operator_selection


def load_plan_parameterization(mod_path: str | Path) -> pb.ParameterGeneration:
    """Loads a `ParameterGeneration` instance from a Python module.

    The target module must contain exactly one operator parameterization instance. The return value of this function is
    the member variable. It is not sufficient to just define a custom parameterization subclass, the module must also
    handle instantiation of the parameterization.

    Warnings
    --------
    In order to determine the parameterization selection, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """

    mod_path = Path(mod_path)
    plan_parameterization = _load_target(mod_path, pb.ParameterGeneration)
    if plan_parameterization is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one PlanParameterization instance.")
    return plan_parameterization


def load_pipeline(mod_path: str | Path) -> pb.OptimizationPipeline:
    """Loads an `OptimizationPipeline` instance from a Python module.

    The target module must contain exactly one pipeline instance. The return value of this function is the member
    variable. It is not sufficient to just define a custom pipeline subclass, the module must also handle instantiation
    of the pipeline.

    Warnings
    --------
    In order to determine the pipeline, the entire module must be executed.
    DO NOT USE THIS FUNCTION WITH UNTRUSTED CODE!!!
    """
    mod_path = Path(mod_path)
    pipeline = _load_target(mod_path, pb.OptimizationPipeline)
    if pipeline is None:
        raise ValueError(f"Module {mod_path} needs to contain exactly one OptimizationPipeline instance.")
    return pipeline
