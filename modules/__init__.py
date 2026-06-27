from .base import Module, ModuleType, MappingType, AxisType
from .input import Input
from .operations import Add, Mult, MatMul, Accumulator
from .activations import Activation
from .learnable import LearnableParameter
from .normalizer import Normalizer
from .structural import Concat, Split, Shift, Transpose
from .pooling import Pooling
from .softmax import SoftMax
from .memory import EMA
from .einsteinAggregator import EinsteinAggregator
from .constant import Constant
from .noise import Noise

ALL_MODULES = [
    EinsteinAggregator,
    Activation,
    LearnableParameter,
    Constant,
    MatMul,
    Add,
    Mult,
    Normalizer,
    Concat,
    Split,
    Pooling,
    Transpose,
    SoftMax,
    Shift,
    Accumulator,
    EMA,
    Noise,
]

UNIFIED_PRESET = [EinsteinAggregator, Activation, LearnableParameter, Constant, Noise]
RICH_PRESET = [MatMul, Add, Activation, LearnableParameter, Normalizer, Mult, Concat,
               Split, Pooling, Transpose, SoftMax, Shift, Accumulator, EMA, Noise]
