import gym
import numpy as np

from .spaces import SquareTiling as Space


class Realm(gym.Env):

    POWER = 0
    ANGEL = 1

    def __init__(
            self,
            marred=False,
            num_agents=1,
            num_powers=1,
            space_length=8,
            episode_length=64,
            seed=None,
    ):
        # Data types
        self._float_dtype = np.float32
        self._int_dtype = np.int32

        # Seeding
        self._np_random = np.random
        if seed is not None:
            self._seed(seed)
        
        assert episode_length > 0
        self._episode_length = episode_length

        # Assign agent types
        self._agent_types = np.zeros(num_agents, dtype=self.int_dtype)
        agent_ids = np.arange(num_agents, dtype=self.int_dtype)
        powers = self.np_random.choice(
            agent_ids, num_powers, replace=False
        )
        angels = np.setdiff1d(agent_ids, powers)
        self._agent_types[powers] = self.POWER
        self._agent_types[angels] = self.ANGEL

        # Create space
        self._space = Space(self.int_dtype, self.np_random, space_length, num_agents)

        # Ensure that there is enough space for all agents and the goal
        assert self.num_agents < self.space.volume

        self._marred = marred

        # Define observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self._action_space = {
            agent_id: gym.spaces.MultiDiscrete(
                tuple([len(action) for action in self.space.actions.values()])
            ) for agent_id in range(self.num_agents)
        }
    
    @property
    def float_dtype(self):
        return self._float_dtype
    
    @property
    def int_dtype(self):
        return self._int_dtype
    
    @property
    def np_random(self):
        return self._np_random
    
    @property
    def episode_length(self):
        return self._episode_length
    
    @property
    def space(self):
        return self._space
    
    @property
    def agent_types(self):
        return self._agent_types

    @property
    def num_agents(self):
        assert self.space.num_agents == len(self.agent_types)
        return self.space.num_agents
    
    @property
    def powers(self):
        return np.where(self.agent_types == self.POWER)[0].astype(self.int_dtype)
    
    @property
    def angels(self):
        return np.where(self.agent_types == self.ANGEL)[0].astype(self.int_dtype)
    
    @property
    def marred(self):
        return self._marred
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def goal_reached(self):
        return np.all(self.space.agent_points == self.goal_point, axis=1)
    
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

    def reset(self):
        # Reset the environment to its initial state
        self.space.reset()

        self.goal_point = self.space.get_unoccupied_point()

        self.time_step = 0

        return self.observations

    def step(self, actions=None):
        # Perform one step in the environment

        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        self.space.step(np.vstack(tuple(actions.values())))

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
