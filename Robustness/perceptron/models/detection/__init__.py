"""Provides class to wrap PaddleHub models in different frameworks
so that they provide a unified API to the benchmarks.
"""
from .pphub import PedestrianDetModel
from .pphub import VehiclesDetModel
from .pphub import SSDVGG16300Model
from .pthub import YOLOv5sModel
from .keras_ssd300 import KerasSSD300Model
