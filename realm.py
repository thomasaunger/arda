import functools
import gym
import numpy as np
import operator

NUM_ORIENTATIONS = 4


class Realm(gym.Env):

    Y = 0
    X = 1

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

    def __init__(self, num_agents=1, max_time_steps=64, grid_shape=(7, 7)):
        # Define your environment variables here

        # Ensure that there are enough grid cells for all agents and the goal
        assert num_agents < functools.reduce(operator.mul, grid_shape)

        self.grid = np.zeros(grid_shape, dtype=np.int8)

        self.max_time_steps = max_time_steps

        self.num_agents = num_agents

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            i: gym.spaces.MultiDiscrete(
                tuple([len(action) for action in Realm.actions.values()])
            ) for i in range(self.num_agents)
        }
    
    def _get_unoccupied_location(self, locations):
        while True:
            location = [np.random.randint(shape) for shape in self.grid.shape]
            if location not in locations:
                return location
    
    def _occupied(self, location):
        return 0 < self.grid[location[0], location[1]]

    def reset(self):
        # Reset the environment to its initial state
        self.grid.fill(0)

        self.agent_locations = []
        for i in range(self.num_agents):
            location = self._get_unoccupied_location(self.agent_locations)
            self.agent_locations.append(location)
        
        self.goal_location = self._get_unoccupied_location(self.agent_locations)
        
        self.agent_locations    = np.array(self.agent_locations)
        self.agent_orientations = np.random.randint(NUM_ORIENTATIONS, size=self.num_agents)

        self.grid[self.agent_locations[:, Realm.Y], self.agent_locations[:, Realm.X]] = 1

        self.goal_reached = False

        self.time_step = 0

        self.last_move_legal = [True]*self.num_agents

    def step(self, actions=None):
        # Perform one step in the environment

        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        for i, action in actions.items():
            match action[Realm.TURN]:
                case Realm.LEFT:
                    self.agent_orientations[i] -= 1
                    self.last_move_legal[i] = True
                case Realm.RIGHT:
                    self.agent_orientations[i] += 1
                    self.last_move_legal[i] = True
            
            self.agent_orientations[i] %= NUM_ORIENTATIONS
            
            match action[Realm.MOVE]:
                case Realm.FORWARD:
                    new_location = self.agent_locations[i].copy()
                    match self.agent_orientations[i]:
                        case Realm.NORTH:
                            new_location[Realm.Y] -= 1
                        case Realm.EAST:
                            new_location[Realm.X] += 1
                        case Realm.SOUTH:
                            new_location[Realm.Y] += 1
                        case Realm.WEST:
                            new_location[Realm.X] -= 1
                    
                    new_location = np.clip(new_location, 0, np.array(self.grid.shape) - 1)

                    if new_location[Realm.Y] == self.goal_location[Realm.Y] and new_location[Realm.X] == self.goal_location[Realm.X]:
                        self.goal_reached = True
                    elif not self._occupied(new_location):
                        self.grid[self.agent_locations[i][Realm.Y], self.agent_locations[i][Realm.X]] = 0
                        self.grid[new_location[Realm.Y], new_location[Realm.X]] = 1
                        self.agent_locations[i] = new_location
                        self.last_move_legal[i] = True
                    else:    
                        self.last_move_legal[i] = False

        self.time_step += 1

        obss = None
        rewards = None
        done = self.goal_reached or self.max_time_steps <= self.time_step
        info = None

        return obss, rewards, done, info
    
    def render(self, *args, **kwargs):
        return 

    def close(self):
        # Clean up resources
        pass

# Register the environment with OpenAI Gym
gym.register(
    id="Realm-v0",
    entry_point="realm:Realm",
)
