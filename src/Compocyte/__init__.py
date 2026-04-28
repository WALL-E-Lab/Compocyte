from . import core
from . import data
from . import pretrained

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"