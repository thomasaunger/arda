import numpy as np

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

from .realm import Realm

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_ = "loc_"
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

        space = self.space.array.copy()

        # add goal point to the space
        space[tuple(self.goal_point.T)] = 1

        for agent_id in range(self.num_agents):
            obss[agent_id] = np.concatenate(
                [
                    self.space.rotate_coordinates(
                        np.array(
                            self.space.agent_points[agent_id],
                            dtype=self.float_dtype,
                        ),
                        self.space.agent_orientations[agent_id]
                    ),
                    self.space.rotate_coordinates(
                        np.array(
                            self.goal_point,
                            dtype=self.float_dtype,
                            ),
                        self.space.agent_orientations[agent_id]
                    ) - self.space.rotate_coordinates(
                        np.array(
                            self.space.agent_points[agent_id],
                            dtype=self.float_dtype,
                        ),
                        self.space.agent_orientations[agent_id]
                    ),
                    self.space.agent_orientations[agent_id].reshape(-1),
                ], dtype=self.float_dtype
            )

        return obss

    @property
    def rewards(self):
        """
        Compute and return the rewards for each agent.
        """
        return self.goal_reached*(1.0 - self.time_step/self.episode_length)
    
    def _coordinate_name(self, coordinate):
        """
        Return the coordinate name for the given coordinate
        """
        return _LOC_ + chr(ord('a') + (self.space.NUM_COORDINATES + ord('x') - 1 - coordinate - ord('a')) % 26)

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
            name="space_length",
            data=self.space.length,
        )
        for coordinate in range(self.space.NUM_COORDINATES):
            data_dict.add_data(
                name=self._coordinate_name(coordinate),
                data=np.ascontiguousarray(self.space.agent_points[:, coordinate]),
                save_copy_and_apply_at_reset=False,
                log_data_across_episode=True,
            )
        data_dict.add_data(
            name=_ORIENTATIONS,
            data=self.space.agent_orientations,
            save_copy_and_apply_at_reset=False,
            log_data_across_episode=True,
        )
        data_dict.add_data(
            name="agent_types",
            data=self.space.agent_types,
        )
        data_dict.add_data(
            name="goal_point",
            data=self.goal_point,
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
                "space_length",
                *[self._coordinate_name(coordinate) for coordinate in range(self.space.NUM_COORDINATES)],
                _ORIENTATIONS,
                "agent_types",
                "goal_point",
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
        else:
            return super().step(actions)
