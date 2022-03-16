"""Provides class to wrap all adversarial criterions
so that attacks has uniform API access.
"""

from .base import Criterion
from .base import CombinedCriteria
from .detection import TargetClassMiss
from .detection import WeightedAP
from .detection import DetObjProbDecrease
