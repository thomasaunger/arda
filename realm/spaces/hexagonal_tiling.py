import numpy as np

from .utils import RegularTiling


class HexagonalTiling(RegularTiling):
    
    _SYMMETRY_ORDER = 6

    _NORTH = 0
    _NORTHEAST = 1
    _SOUTHEAST = 2
    _SOUTH = 3
    _SOUTHWEST  = 4
    _NORTHWEST = 5

    def __init__(self, int_dtype, np_random, radius, num_agents, action_space):
        super().__init__(int_dtype, np_random, radius, num_agents, action_space)

    @property
    def NORTH(self):
        return self._NORTH
    
    @property
    def NORTHEAST(self):
        return self._NORTHEAST
    
    @property
    def SOUTHEAST(self):
        return self._SOUTHEAST
    
    @property
    def SOUTH(self):
        return self._SOUTH
    
    @property
    def NORTHWEST(self):
        return self._NORTHWEST
    
    @property
    def SOUTHWEST(self):
        return self._SOUTHWEST
    
    @property
    def principal_orientation(self):
        # [z, y, x] = [s, r, q]
        # https://www.redblobgames.com/grids/hexagons/
        return np.array([1, -1, 0], dtype=self.int_dtype)
    
    @property
    def R(self):
        return self._R(
            np.array(
                [ # [ z,  y,  x]
                    [ 0,  0, -1], # z = -x
                    [-1,  0,  0], # y = -z
                    [ 0, -1,  0]  # x = -y
                ], dtype=self.int_dtype
            )
        )

    @property
    def volume(self):
        return self.length**2 - self.radius*(self.radius + 1)
    
    def _random_point(self):
        point = [self.np_random.randint(shape, dtype=self.int_dtype) for shape in self.array.shape[1:]]
        while sum(point) < self.radius or 3*self.radius < sum(point):
            point = [self.np_random.randint(shape, dtype=self.int_dtype) for shape in self.array.shape[1:]]
        return np.array([3*self.radius - sum(point), *point], dtype=self.int_dtype)
