import numpy as np

from .space import Space


class RegularTiling(Space):
    
    def __init__(self, int_dtype, np_random, length, agent_class, num_agents, num_powers):
        super().__init__(int_dtype, np_random, length, agent_class, num_agents, num_powers)
        
        self._array = np.zeros((self.length,)*self.NUM_COORDINATES, dtype=self.int_dtype)

        self._agent_points = np.zeros((self.num_agents, self.NUM_COORDINATES), dtype=self.int_dtype)
    
    @property
    def SYMMETRY_ORDER(self):
        return self._SYMMETRY_ORDER
    
    @property
    def NUM_COORDINATES(self):
        return self._NUM_COORDINATES
    
    @property
    def array(self):
        return self._array
    
    @property
    def agent_points(self):
        return self._agent_points

    @property
    def agent_orientations(self):
        return self._agent_orientations

    def _random_coordinates(self):
        return np.array([self.np_random.randint(shape, dtype=self.int_dtype) for shape in self.array.shape], dtype=self.int_dtype)
    
    def get_unoccupied_point(self):
        while True:
            point = self._random_coordinates()
            if not np.any(np.all(point == self.agent_points, axis=1)):
                return point
    
    def occupied(self, point):
        return 0 < self.array[tuple(point.T)]

    def reset(self):
        self.array.fill(0)

        for i in range(self.num_agents):
            while True:
                point = self._random_coordinates()
                if not np.any(np.all(point == self.agent_points[:i], axis=1)):
                    break
            self._agent_points[i] = point

        self.array[tuple(self.agent_points.T)] = np.arange(self.num_agents, dtype=self.int_dtype) + 2

        self._agent_orientations = self.np_random.randint(self.SYMMETRY_ORDER, size=self.num_agents, dtype=self.int_dtype)
