import numpy as np

from .utils import RegularTiling


class SquareTiling(RegularTiling):
    
    SYMMETRY_ORDER = 4
    NUM_COORDINATES = 2

    def __init__(self, length, int_dtype):
        super().__init__(length, int_dtype)

        self._surface = np.zeros((self.length,)*self.NUM_COORDINATES, dtype=self.int_dtype)

    @property
    def volume(self):
        return self.length**2
