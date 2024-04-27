import functools
import gym
import numpy as np
import operator

from .utils import Agent
from .utils.tilings import SquareTiling


class Realm(gym.Env):

    COORDINATE_Y = 0
    COORDINATE_X = 1

    NORTH = 0
    EAST  = 1
    SOUTH = 2
    WEST  = 3

    TURN = 0
    MOVE = 1

    NONE = 0
    
    LEFT = 1
    RIGHT = 2

    FORWARD = 1

    POWER = 0
    ANGEL = 1

    actions = {
        TURN: {
            "NONE":  NONE,
            "LEFT":  LEFT,
            "RIGHT": RIGHT,
        },
        MOVE: {
            "NONE":    NONE,
            "FORWARD": FORWARD,
        },
    }

    def __init__(
            self,
            marred=False,
            num_agents=1,
            num_powers=1,
            surface_length=8,
            episode_length=64,
            seed=None,
    ):
        # Data types
        self.float_dtype = np.float32
        self.int_dtype = np.int32

        # Create surface
        self.surface = SquareTiling(surface_length, self.int_dtype)

        # Ensure that there is enough space for all agents and the goal
        assert num_agents < self.surface.volume

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self._seed(seed)

        assert episode_length > 0
        self.episode_length = episode_length

        self.num_agents = num_agents

        self.num_powers = num_powers

        # Starting powers
        powers = self.np_random.choice(
            np.arange(self.num_agents), self.num_powers, replace=False
        )

        self.agent_types = {}
        self.powers = {}
        self.angels = {}
        for agent_id in range(self.num_agents):
            if agent_id in set(powers):
                self.agent_types[agent_id] = Realm.POWER
                self.powers[agent_id] = True
            else:
                self.agent_types[agent_id] = Realm.ANGEL
                self.angels[agent_id] = True

        # These will be set during reset (see below)
        self.time_step = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: gym.spaces.MultiDiscrete(
                tuple([len(action) for action in Realm.actions.values()])
            ) for agent_id in range(self.num_agents)
        }

        self.marred = marred
    
    @property
    def observations(self):
        return None

    @property
    def rewards(self):
        return None
    
    def _seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random.seed(seed)
        return [seed]
    
    def _get_unoccupied_location(self, locations):
        while True:
            location = [np.random.randint(shape, dtype=self.int_dtype) for shape in self.surface._surface.shape]
            if location not in locations:
                return location
    
    def _occupied(self, location):
        return 0 < self.surface._surface[location[0], location[1]]

    def reset(self):
        # Reset the environment to its initial state
        self.surface._surface.fill(0)

        self.agent_locations = []
        for _ in range(self.num_agents):
            location = self._get_unoccupied_location(self.agent_locations)
            self.agent_locations.append(location)
        
        self.goal_location = self._get_unoccupied_location(self.agent_locations)
        
        self.agent_locations    = np.array(self.agent_locations, dtype=self.int_dtype)
        self.agent_orientations = np.random.randint(self.surface.symmetry_order, size=self.num_agents, dtype=self.int_dtype)

        self.surface._surface[self.agent_locations[:, Realm.COORDINATE_Y], self.agent_locations[:, Realm.COORDINATE_X]] = np.arange(self.num_agents, dtype=self.int_dtype) + 2

        self.goal_reached = np.array([False]*self.num_agents, dtype=self.int_dtype)

        self.time_step = 0

        self.global_state = {}

        self.last_move_legal = [True]*self.num_agents

        self.first_action = [True]*self.num_agents

        return self.observations

    def step(self, actions=None):
        # Perform one step in the environment

        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        for agent_id, action in actions.items():
            if 0 < action[Realm.MOVE]:
                match action[Realm.MOVE]:
                    case Realm.FORWARD:
                        new_location = self.agent_locations[agent_id].copy()
                        match self.agent_orientations[agent_id]:
                            case Realm.NORTH:
                                new_location[Realm.COORDINATE_Y] -= 1
                                self.first_action[agent_id] = False
                            case Realm.EAST:
                                new_location[Realm.COORDINATE_X] += 1
                                self.first_action[agent_id] = False
                            case Realm.SOUTH:
                                new_location[Realm.COORDINATE_Y] += 1
                                self.first_action[agent_id] = False
                            case Realm.WEST:
                                new_location[Realm.COORDINATE_X] -= 1
                                self.first_action[agent_id] = False
                        
                        new_location = np.clip(new_location, 0, np.array(self.surface._surface.shape) - 1, dtype=self.int_dtype)

                        if new_location[Realm.COORDINATE_Y] == self.goal_location[Realm.COORDINATE_Y] and new_location[Realm.COORDINATE_X] == self.goal_location[Realm.COORDINATE_X]:
                            self.goal_reached[agent_id] = True
                        elif True:  # not self._occupied(new_location):
                            self.surface._surface[self.agent_locations[agent_id][Realm.COORDINATE_Y], self.agent_locations[agent_id][Realm.COORDINATE_X]] = 0
                            self.surface._surface[new_location[Realm.COORDINATE_Y], new_location[Realm.COORDINATE_X]] = agent_id + 2
                            self.agent_locations[agent_id] = new_location
                            self.last_move_legal[agent_id] = True
                        else:    
                            self.last_move_legal[agent_id] = False
            else:
                match action[Realm.TURN]:
                    case Realm.LEFT:
                        self.agent_orientations[agent_id] -= 1
                        self.last_move_legal[agent_id] = True
                        self.first_action[agent_id] = False
                    case Realm.RIGHT:
                        self.agent_orientations[agent_id] += 1
                        self.last_move_legal[agent_id] = True
                        self.first_action[agent_id] = False
                
                self.agent_orientations[agent_id] %= self.surface.symmetry_order

        obss = self.observations
        rewards = self.rewards
        done = any(self.goal_reached) or self.episode_length <= self.time_step
        info = None

        self.time_step += 1

        return obss, rewards, done, info

    def close(self):
        # Clean up resources
        pass

# Register the environment with OpenAI Gym
gym.register(
    id="Realm-v0",
    entry_point="realm:Realm",
)
