import numpy as np

from . import Agent


class Registry:
    
    def __init__(self, num_agents, num_powers, np_random, int_dtype):
        self.np_random = np_random
        self.int_dtype = int_dtype

        self._agent_types = np.zeros(num_agents, dtype=self.int_dtype)

        agent_ids = np.arange(num_agents, dtype=self.int_dtype)
        powers = self.np_random.choice(
            agent_ids, num_powers, replace=False
        )
        angels = np.setdiff1d(agent_ids, self.powers)

        self._agent_types[powers] = Agent.POWER
        self._agent_types[angels] = Agent.ANGEL

    @property
    def agent_types(self):
        return self._agent_types

    @property
    def num_agents(self):
        return len(self.agent_types)

    @property
    def powers(self):
        return np.where(self._agent_types == Agent.POWER)[0].astype(self.int_dtype)
    
    @property
    def angels(self):
        return np.where(self._agent_types == Agent.ANGEL)[0].astype(self.int_dtype)
