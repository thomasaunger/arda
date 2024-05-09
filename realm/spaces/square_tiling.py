import numpy as np

from .utils import RegularTiling


class SquareTiling(RegularTiling):
    
    _SYMMETRY_ORDER = 4
    _NUM_COORDINATES = 2

    NORTH = 0
    EAST  = 1
    SOUTH = 2
    WEST  = 3

    @property
    def volume(self):
        return self.length**2

    def rotate_coordinates(self, coordinates, orientation):
        """
        Rotate the coordinates based on orientation
        """
        # TODO: implement using matrix multiplication
        k = self.length - 1
        y = coordinates[0]
        x = coordinates[1]
        match orientation:
            case self.NORTH:
                return coordinates.copy()
            case self.EAST:
                return np.array([k - x, y    ], dtype=self.int_dtype)
            case self.SOUTH:
                return np.array([k - y, k - x], dtype=self.int_dtype)
            case self.WEST:
                return np.array([x,     k - y], dtype=self.int_dtype)

    def delta(self, orientation):
        """
        Return one-step delta vector based on orientation
        """
        match orientation:
            case self.NORTH:
                return np.array([-1, 0], dtype=self.int_dtype)
            case self.EAST:
                return np.array([ 0, 1], dtype=self.int_dtype)
            case self.SOUTH:
                return np.array([ 1, 0], dtype=self.int_dtype)
            case self.WEST:
                return np.array([0, -1], dtype=self.int_dtype)
