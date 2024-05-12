import numpy as np

from .utils import RegularTiling


class SquareTiling(RegularTiling):
    
    _SYMMETRY_ORDER = 4
    _NUM_COORDINATES = 2

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
    def volume(self):
        return self.length**2

    def rotate_coordinates(self, coordinates, orientation):
        """
        Rotate the coordinates based on orientation
        """        
        new_coordinates = coordinates.copy()
        for _ in range(orientation):
            new_coordinates = np.array([self.length - 1, 0], dtype=self.int_dtype) + np.array(
                [
                    [0, -1],
                    [1,  0]
                ], dtype=self.int_dtype
            ).dot(new_coordinates)
        return new_coordinates

    def delta(self, orientation):
        """
        Return one-step delta vector based on orientation
        """
        delta = np.array([-1, 0])
        for _ in range(orientation):
            delta = np.array(
                [
                    [ 0, 1],
                    [-1, 0]
                ], dtype=self.int_dtype
            ).dot(delta)
        return delta
