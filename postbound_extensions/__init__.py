"""PostBOUND Extensions - A curated collection of novel ideas in query optimization."""

from importlib.metadata import PackageNotFoundError, version

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

try:
    __version__ = version("Optimizer-Playground")
except PackageNotFoundError:
    __version__ = "<development>"
