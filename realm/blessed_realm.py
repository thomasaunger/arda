import numpy as np

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

from .realm import Realm

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_Y = "loc_y"
_LOC_X = "loc_x"
_ORIENTATIONS = "orientations"


class BlessedRealm(Realm, CUDAEnvironmentContext):
    
    def __init__(
            self,
            *args,
            env_backend="cpu",
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # These will also be set via the env_wrapper
        self.env_backend = env_backend

    name = "BlessedRealm"

    @property
    def observations(self):
        """
        Generate and return the observations for every agent.
        """
        obss = {}

        grid = self.grid.copy()

        # add goal location to the grid
        grid[self.goal_location[Realm.Y], self.goal_location[Realm.X]] = 1

        for i in range(self.num_agents):
            obss[i] = np.concatenate(
                [
                    np.array([i + 2], dtype=self.int_dtype),
                    np.rot90(grid, k=self.agent_orientations[i]).reshape(-1),
                    (self.agent_orientations - self.agent_orientations[i]) % Realm.NUM_ORIENTATIONS,
                ], dtype=self.int_dtype
            )

        return obss

    @property
    def rewards(self):
        """
        Compute and return the rewards for each agent.
        """
        return self.goal_reached*(1.0 - self.time_step/self.episode_length)

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        data_dict.add_data(
            name="marred",
            data=self.marred,
        )
        data_dict.add_data(
            name="grid_length",
            data=self.grid_length,
        )
        data_dict.add_data(
            name=_LOC_Y,
            data=self.agent_locations[:, Realm.Y],
        )
        data_dict.add_data(
            name=_LOC_X,
            data=self.agent_locations[:, Realm.X],
        )
        data_dict.add_data(
            name=_ORIENTATIONS,
            data=self.agent_orientations,
        )
        return data_dict
    
    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        if not self.env_backend == "cpu":
            # CUDA version of step()

            args = [
                "marred",
                "grid_length",
                _LOC_Y,
                _LOC_X,
                _ORIENTATIONS,
                _OBSERVATIONS,
                _ACTIONS,
                _REWARDS,
                "_done_",
                "_timestep_",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
            ]

            if self.env_backend == "pycuda":
                self.cuda_step(
                    *self.cuda_step_function_feed(args),
                    block=self.cuda_function_manager.block,
                    grid=self.cuda_function_manager.grid,
                )

            self.time_step += 1

            return None
        else:
            obss, rewards, done, info = super().step(actions)

            return obss, rewards, done, info