import numpy as np

from ... import Agent


class Space:
    
    def __init__(self, int_dtype, np_random, length, num_agents, num_powers):
        assert length >= 0, "Space length must be non-negative"

        self._int_dtype = int_dtype
        self._np_random = np_random
        self._length    = length

        self._agent_types = np.zeros(num_agents, dtype=self._int_dtype)

        agent_ids = np.arange(num_agents, dtype=self.int_dtype)
        powers = self.np_random.choice(
            agent_ids, num_powers, replace=False
        )
        angels = np.setdiff1d(agent_ids, self.powers)

        self._agent_types[powers] = Agent.POWER
        self._agent_types[angels] = Agent.ANGEL
    
    @property
    def int_dtype(self):
        return self._int_dtype
    
    @property
    def np_random(self):
        return self._np_random
    
    @property
    def length(self):
        return self._length
    
    @property
    def agent_types(self):
        return self._agent_types

    @property
    def num_agents(self):
        return len(self.agent_types)

    @property
    def powers(self):
        return np.where(self.agent_types == Agent.POWER)[0].astype(self.int_dtype)
    
    @property
    def angels(self):
        return np.where(self.agent_types == Agent.ANGEL)[0].astype(self.int_dtype)
