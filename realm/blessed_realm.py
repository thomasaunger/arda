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
        grid[self.goal_location[Realm.COORDINATE_Y], self.goal_location[Realm.COORDINATE_X]] = 1

        for agent_id in range(self.num_agents):
            obss[agent_id] = np.concatenate(
                [
                    np.array([agent_id + 2], dtype=self.float_dtype),
                    np.rot90(grid, k=self.agent_orientations[agent_id]).reshape(-1),
                    (self.agent_orientations - self.agent_orientations[agent_id]) % Realm.NUM_ORIENTATIONS,
                ], dtype=self.float_dtype
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
            data=np.ascontiguousarray(self.agent_locations[:, Realm.COORDINATE_Y]),
            save_copy_and_apply_at_reset=False,
            log_data_across_episode=True,
        )
        data_dict.add_data(
            name=_LOC_X,
            data=np.ascontiguousarray(self.agent_locations[:, Realm.COORDINATE_X]),
            save_copy_and_apply_at_reset=False,
            log_data_across_episode=True,
        )
        data_dict.add_data(
            name=_ORIENTATIONS,
            data=self.agent_orientations,
            save_copy_and_apply_at_reset=False,
            log_data_across_episode=True,
        )
        data_dict.add_data(
            name="agent_types",
            data=np.array([self.agent_types[agent_id] for agent_id in range(self.num_agents)], dtype=self.int_dtype),
        )
        data_dict.add_data(
            name="goal_location",
            data=self.goal_location,
            save_copy_and_apply_at_reset=False,
            log_data_across_episode=True,
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
                "agent_types",
                "goal_location",
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
            return super().step(actions)
