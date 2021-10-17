"""Provides class to wrap existing models in different frameworks
so that they provide a unified API to the benchmarks.
"""

from .kerasmodel import KerasModel
from .pytorch import PyTorchModel
from .paddle import PaddleModel
