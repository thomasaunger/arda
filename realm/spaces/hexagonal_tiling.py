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
        super().__init__(int_dtype, np_random, 2*radius + 1, num_agents, action_space)

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
    def radius(self):
        return self.length//2
    
    @property
    def center(self):
        return np.array([-2*self.radius, self.radius, self.radius], dtype=self.int_dtype)
    
    @property
    def principal_orientation(self):
        # [z, y, x] = [s, r, q]
        # https://www.redblobgames.com/grids/hexagons/
        return np.array([1, -1, 0], dtype=self.int_dtype)
    
    @property
    def R(self):
        return super().R(
            np.array(
                [ # [ s,  r,  q]
                    [ 0,  0, -1], # s = -q
                    [-1,  0,  0], # r = -s
                    [ 0, -1,  0]  # q = -r
                ], dtype=self.int_dtype
            )
        )

    @property
    def volume(self):
        return self.length**2

    def rotate_coordinates(self, coordinates, orientation):
        """
        Rotate the coordinates based on orientation
        """
        return self.L.pow(orientation).dot(coordinates - self.center) + self.center
