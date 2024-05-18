import numpy as np

from .space import Space
from .rotation import Rotation


class RegularTiling(Space):
    
    def __init__(self, int_dtype, np_random, length, num_agents):
        super().__init__(int_dtype, np_random, length)

        self._num_agents = num_agents

        self._dims = len(self.principal_orientation)
        
        self._array = np.zeros((self.length,)*self.dims, dtype=self.int_dtype)

        self._agent_points = np.zeros((self.num_agents, self.dims), dtype=self.int_dtype)
    
    @property
    def SYMMETRY_ORDER(self):
        return self._SYMMETRY_ORDER
    
    @property
    def num_agents(self):
        return self._num_agents
    
    @property
    def dims(self):
        return self._dims
    
    @property
    def array(self):
        return self._array
    
    @property
    def agent_points(self):
        return self._agent_points

    @property
    def agent_orientations(self):
        return self._agent_orientations
    
    @property
    def L(self):
        return Rotation(self.R.pow(-1), self.SYMMETRY_ORDER)
    
    def R(self, matrix):
        return Rotation(matrix, self.SYMMETRY_ORDER)

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

    def step(self, actions):
        if 0 < (i_movers := np.where(
            actions.T[self.MOVE] == self.FORWARD
        )[0]).size:
            self.array[tuple(self.agent_points[i_movers].T)] = 0
            self.agent_points[i_movers] = np.clip(
                self.agent_points[i_movers] + np.array(
                    [
                        self.R.pow(agent_orientation).dot(self.principal_orientation) for agent_orientation in self.agent_orientations[i_movers]
                    ]
                ),
                0,
                np.array(self.array.shape) - 1,
                dtype=self.int_dtype
            )
            self.array[tuple(self.agent_points[i_movers].T)] = i_movers + 2

        if 0 < (i_lefters := np.where(
            actions.T[self.MOVE] == self.NONE and
            actions.T[self.TURN] == self.LEFT
        )[0]).size:
            self.agent_orientations[i_lefters] -= 1
            self.agent_orientations[i_lefters] %= self.SYMMETRY_ORDER

        if 0 < (i_righters := np.where(
            actions.T[self.MOVE] == self.NONE and
            actions.T[self.TURN] == self.RIGHT
        )[0]).size:
            self.agent_orientations[i_righters] += 1
            self.agent_orientations[i_righters] %= self.SYMMETRY_ORDER
