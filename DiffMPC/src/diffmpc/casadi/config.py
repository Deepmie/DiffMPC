from dataclasses import dataclass
from numpy import ndarray
from typing import Optional

@dataclass
class BaseConfig:
    state_dim: int   = 3
    control_dim: int = 4
    T: int           = 5
    # cost parameter
    C: Optional[ndarray] = None
    c: Optional[ndarray] = None

    # dynamic parameter
    F: Optional[ndarray] = None
    f: Optional[ndarray] = None

    def get_parameters(self):
        return self.C, self.c, self.F, self.f