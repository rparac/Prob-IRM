try:
    from .alergia import AlergiaLearner
except ImportError:
    AlergiaLearner = type(None)
try:
    from .dafsa import DAFSALearner
except ImportError:
    DAFSALearner = type(None)
try:
    from .ilasp import ILASPLearner
except ImportError:
    ILASPLearner = type(None)
from .learner import RMLearner
try:
    from .samp2symb import S2SLearner
except ImportError:
    S2SLearner = type(None)
