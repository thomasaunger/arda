import numpy as np


class Rotation(np.ndarray):

    def __new__(cls, R, symmetry_order, *args, **kwargs):
        return np.asarray(R).view(cls)
    
    def __init__(self, R, symmetry_order, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._SYMMETRY_ORDER = symmetry_order
    
    @property
    def SYMMETRY_ORDER(self):
        return self._SYMMETRY_ORDER

    def pow(self, n):
        return np.linalg.matrix_power(self, n % self.SYMMETRY_ORDER)
