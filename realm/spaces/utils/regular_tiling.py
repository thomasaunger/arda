import numpy as np

from .space import Space
from .rotation import Rotation


class RegularTiling(Space):
    
    def __init__(self, int_dtype, np_random, length, agent_class, num_agents, num_powers):
        super().__init__(int_dtype, np_random, length, agent_class, num_agents, num_powers)

        self._dims = len(self.R)
        
        self._array = np.zeros((self.length,)*self.dims, dtype=self.int_dtype)

        self._agent_points = np.zeros((self.num_agents, self.dims), dtype=self.int_dtype)
    
    @property
    def SYMMETRY_ORDER(self):
        return self._SYMMETRY_ORDER
    
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
        for agent_id, action in enumerate(actions):
            if 0 < action[self.agent_class.MOVE]:
                match action[self.agent_class.MOVE]:
                    case self.agent_class.FORWARD:
                        self.array[tuple(self.agent_points[agent_id].T)] = 0
                        orientation = self.R.pow(self.agent_orientations[agent_id]).dot(self.principal_orientation)
                        self.agent_points[agent_id] = np.clip(self.agent_points[agent_id] + orientation, 0, np.array(self.array.shape) - 1, dtype=self.int_dtype)
                        self.array[tuple(self.agent_points[agent_id].T)] = agent_id + 2
            else:
                match action[self.agent_class.TURN]:
                    case self.agent_class.LEFT:
                        self.agent_orientations[agent_id] -= 1
                    case self.agent_class.RIGHT:
                        self.agent_orientations[agent_id] += 1

            self.agent_orientations[agent_id] %= self.SYMMETRY_ORDER