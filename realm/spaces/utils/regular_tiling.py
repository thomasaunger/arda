import numpy as np

from .space import Space
from .rotation import Rotation


class RegularTiling(Space):
    
    def __init__(self, int_dtype, np_random, radius, num_agents, action_space):
        super().__init__(int_dtype, np_random, radius)

        self._num_agents = num_agents

        self._action_space = action_space

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
    def action_space(self):
        return self._action_space
    
    @property
    def dims(self):
        return self._dims
    
    @property
    def array(self):
        return self._array
    
    @property
    def center(self):
        return (np.array(self.array.shape, dtype=self.int_dtype) - 1)//2
    
    @property
    def agent_points(self):
        return self._agent_points

    @property
    def agent_orientations(self):
        return self._agent_orientations
    
    @property
    def L(self):
        return Rotation(self.R.pow(-1), self.SYMMETRY_ORDER)
    
    def _R(self, matrix):
        return Rotation(matrix, self.SYMMETRY_ORDER)
    
    def _validate_points(self, points):
        return np.where(
            np.all(
                np.clip(
                    points,
                    0,
                    np.array(self.array.shape) - 1,
                    dtype=self.int_dtype
                ) == points, axis=1
            )
        )[0].astype(self.int_dtype)

    def rotate_coordinates(self, coordinates, orientation):
        """
        Rotate the coordinates based on orientation
        """
        return self.L.pow(orientation).dot(coordinates - self.center) + self.center
        
    def get_unoccupied_point(self):
        while True:
            point = self._random_point()
            if not np.any(np.all(point == self.agent_points, axis=1)):
                return point

    def reset(self):
        self.array.fill(0)

        for i in range(self.num_agents):
            while True:
                point = self._random_point()
                if not np.any(np.all(point == self.agent_points[:i], axis=1)):
                    break
            self._agent_points[i] = point

        self.array[tuple(self.agent_points.T)] = np.arange(self.num_agents, dtype=self.int_dtype) + 2

        self._agent_orientations = self.np_random.randint(self.SYMMETRY_ORDER, size=self.num_agents, dtype=self.int_dtype)

    def step(self, actions):
        if 0 < (i_movers := np.where(
            actions.T[self.action_space.MOVE] == self.action_space.MOVE.FORWARD
        )[0].astype(self.int_dtype)).size:
            new_agent_points = self.agent_points[i_movers] + np.array(
                [
                    self.R.pow(agent_orientation).dot(self.principal_orientation) for agent_orientation in self.agent_orientations[i_movers]
                ]
            )
            if 0 < (i_movers_validated := self._validate_points(new_agent_points)).size:
                self.array[tuple(self.agent_points[i_movers[i_movers_validated]].T)] = 0
                self.agent_points[i_movers[i_movers_validated]] = new_agent_points[i_movers_validated]
                self.array[tuple(self.agent_points[i_movers[i_movers_validated]].T)] = i_movers[i_movers_validated] + 2

        if 0 < (i_lefters := np.intersect1d(
                np.where(
                    actions.T[self.action_space.MOVE] == self.action_space.MOVE.NONE
                )[0],
                np.where(
                    actions.T[self.action_space.TURN] == self.action_space.TURN.LEFT
                )[0]
            ).astype(self.int_dtype)).size:
            self.agent_orientations[i_lefters] -= 1
            self.agent_orientations[i_lefters] %= self.SYMMETRY_ORDER

        if 0 < (i_righters := np.intersect1d(
                np.where(
                    actions.T[self.action_space.MOVE] == self.action_space.MOVE.NONE
                )[0],
                np.where(
                    actions.T[self.action_space.TURN] == self.action_space.TURN.RIGHT
                )[0]
            ).astype(self.int_dtype)).size:
            self.agent_orientations[i_righters] += 1
            self.agent_orientations[i_righters] %= self.SYMMETRY_ORDER
