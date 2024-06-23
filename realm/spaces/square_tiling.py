import numpy as np

from .utils import RegularTiling


class SquareTiling(RegularTiling):
    
    _SYMMETRY_ORDER = 4

    _NORTH = 0
    _EAST  = 1
    _SOUTH = 2
    _WEST  = 3

    @property
    def NORTH(self):
        return self._NORTH
    
    @property
    def EAST(self):
        return self._EAST
    
    @property
    def SOUTH(self):
        return self._SOUTH
    
    @property
    def WEST(self):
        return self._WEST
    
    @property
    def principal_orientation(self):
        return np.array([-1, 0], dtype=self.int_dtype)
    
    @property
    def R(self):
        return self._R(
            np.array(
                [
                    [ 0, 1],
                    [-1, 0]
                ], dtype=self.int_dtype
            )
        )

    @property
    def volume(self):
        return self.length**2
    
    def _random_point(self):
        return np.array([self.np_random.randint(shape, dtype=self.int_dtype) for shape in self.array.shape], dtype=self.int_dtype)
