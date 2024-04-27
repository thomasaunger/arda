import numpy as np

from .utils import RegularTiling


class SquareTiling(RegularTiling):
    
    def __init__(self, length, int_dtype):
        super().__init__(length, int_dtype)

        self.symmetry_order = 4

        self._surface = np.zeros((self.length, self.length), dtype=self.int_dtype)

    @property
    def volume(self):
        return self.length*self.length
