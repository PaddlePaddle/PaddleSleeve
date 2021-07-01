"""Provides class to wrap all adversarial criterions
so that attacks has uniform API access.
"""

from .base import Criterion
from .base import CombinedCriteria
from .classification import Misclassification
from .classification import ConfidentMisclassification
from .classification import TopKMisclassification
from .classification import TargetClass
from .classification import OriginalClassProbability
from .classification import TargetClassProbability
from .classification import MisclassificationAntiPorn
from .classification import MisclassificationSafeSearch
from .detection import TargetClassMiss
from .detection import TargetClassMissGoogle
from .detection import WeightedAP
