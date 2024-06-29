__constant__ int DIMS = 3;

__constant__ int SYMMETRY_ORDER = 6;

__constant__ int NORTH     = 0;
__constant__ int NORTHEAST = 1;
__constant__ int SOUTHEAST = 2;
__constant__ int SOUTH     = 3;
__constant__ int SOUTHWEST = 4;
__constant__ int NORTHWEST = 5;

__constant__ int LEFT  = 1;
__constant__ int RIGHT = 2;

__constant__ int FORWARD = 1;

extern "C" {
  // Device helper function to rotate coordinates
  __device__ void RotateCoordinates(
    int kRadius,
    int orientation,
    int * loc_z,
    int * loc_y,
    int * loc_x
  ) {
    int loc_z_tmp = *loc_z - kRadius;
    int loc_y_tmp = *loc_y - kRadius;
    int loc_x_tmp = *loc_x - kRadius;
    if        (orientation == NORTH    ) {
      *loc_z = kRadius + loc_z_tmp;
      *loc_y = kRadius + loc_y_tmp;
      *loc_x = kRadius + loc_x_tmp;
    } else if (orientation == NORTHEAST) {
      *loc_z = kRadius - loc_y_tmp;
      *loc_y = kRadius - loc_x_tmp;
      *loc_x = kRadius - loc_z_tmp;
    } else if (orientation == SOUTHEAST) {
      *loc_z = kRadius + loc_x_tmp;
      *loc_y = kRadius + loc_z_tmp;
      *loc_x = kRadius + loc_y_tmp;
    } else if (orientation == SOUTH    ) {
      *loc_z = kRadius - loc_z_tmp;
      *loc_y = kRadius - loc_y_tmp;
      *loc_x = kRadius - loc_x_tmp;
    } else if (orientation == SOUTHWEST) {
      *loc_z = kRadius + loc_y_tmp;
      *loc_y = kRadius + loc_x_tmp;
      *loc_x = kRadius + loc_z_tmp;
    } else if (orientation == NORTHWEST) {
      *loc_z = kRadius - loc_x_tmp;
      *loc_y = kRadius - loc_z_tmp;
      *loc_x = kRadius - loc_y_tmp;
    }
  }

  // Device helper function to check whether a point is within bounds
  __device__ bool PointIsWithinBounds(
    int kRadius,
    int kSpaceLength,
    int loc_z,
    int loc_y,
    int loc_x
  ) {
    return (0 <= loc_z && loc_z < kSpaceLength &&
            0 <= loc_y && loc_y < kSpaceLength &&
            0 <= loc_x && loc_x < kSpaceLength &&
            loc_z + loc_y + loc_x == DIMS*kRadius);
  }

  // Device helper function to check whether a point is occupied
  __device__ bool PointIsOccupied(
    int * loc_z_arr,
    int * loc_y_arr,
    int * loc_x_arr,
    int kNumAgents,
    int kEnvId,
    int kThisAgentId,
    int loc_z,
    int loc_y,
    int loc_x
  ) {
    for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
      if (kAgentId != kThisAgentId) {
        int kAgentIdx = kEnvId * kNumAgents + kAgentId;
        if (loc_z_arr[kAgentIdx] == loc_z && loc_y_arr[kAgentIdx] == loc_y && loc_x_arr[kAgentIdx] == loc_x) {
          return true;
        }
      }
    }
    return false;
  }

  // Device helper function to generate an unoccupied point
  __device__ void GenerateUnoccupiedPoint(
    int kRadius,
    int kSpaceLength,
    int * loc_z_arr,
    int * loc_y_arr,
    int * loc_x_arr,
    int kNumAgents,
    int kEnvId,
    int * loc_z,
    int * loc_y,
    int * loc_x
  ) {
    // Use last agent's state to generate a random point
    curandState_t* state = states[kEnvId * kNumAgents + kNumAgents - 1];
    do {
      // Generate random coordinates from uniform distribution over [0, kSpaceLength - 1]
      *loc_y = kSpaceLength * (1.0 - curand_uniform(state));
      *loc_x = kSpaceLength * (1.0 - curand_uniform(state));
      *loc_z = DIMS*kRadius - *loc_y - *loc_x;
    } while (!PointIsWithinBounds(
      kRadius,
      kSpaceLength,
      *loc_z,
      *loc_y,
      *loc_x
    ) || PointIsOccupied(
      loc_z_arr,
      loc_y_arr,
      loc_x_arr,
      kNumAgents,
      kEnvId,
      -1,
      *loc_z,
      *loc_y,
      *loc_x
    ));
  }

  // Device helper function to compute rewards
  __device__ void CudaBlessedRealmComputeReward(
    int * loc_z_arr,
    int * loc_y_arr,
    int * loc_x_arr,
    int * goal_point_arr,
    float * rewards_arr,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    if (kThisAgentId < kNumAgents) {
      // Initialize rewards
      rewards_arr[kThisAgentArrayIdx] = 0.0;

      // Check whether the agent has reached the goal
      if (loc_z_arr[kThisAgentArrayIdx] == goal_point_arr[kEnvId * DIMS    ] &&
          loc_y_arr[kThisAgentArrayIdx] == goal_point_arr[kEnvId * DIMS + 1] &&
          loc_x_arr[kThisAgentArrayIdx] == goal_point_arr[kEnvId * DIMS + 2]) {
        rewards_arr[kThisAgentArrayIdx] = 1.0 * (1.0 - env_timestep_arr[kEnvId] / float(kEpisodeLength));
        // done_arr[kEnvId] = 1;
      }

      // Use only last agent's thread to check whether the maximum number of timesteps has been reached
      if (kThisAgentId == kNumAgents - 1) {
        if (env_timestep_arr[kEnvId] == kEpisodeLength) {
            done_arr[kEnvId] = 1;
        }
      }
    }
  }

  // Device helper function to generate observation
  __device__ void CudaBlessedRealmGenerateObservation(
    int kRadius,
    int kSpaceLength,
    int * loc_z_arr,
    int * loc_y_arr,
    int * loc_x_arr,
    int * orientation_arr,
    int * goal_point_arr,
    float * obs_arr,
    int * done_arr,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    if (kThisAgentId < kNumAgents) {
      // obs shape is (num_envs, kNumAgents, n)
      const int n = 7;
      const int kThisAgentIdxOffset = (kEnvId * kNumAgents + kThisAgentId) * n;

      // Initialize obs
      for (int i = 0; i < n; i++) {
        obs_arr[kThisAgentIdxOffset + i] = 0.0;
      }

      int kAgentIdx;
      int loc_z;
      int loc_y;
      int loc_x;
      if (done_arr[kEnvId]) {
        // Reinitialize agent points
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          loc_z_arr[kAgentIdx] = -1;
          loc_y_arr[kAgentIdx] = -1;
          loc_x_arr[kAgentIdx] = -1;
        }

        // Reset agent points
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          GenerateUnoccupiedPoint(
            kRadius,
            kSpaceLength,
            loc_z_arr,
            loc_y_arr,
            loc_x_arr,
            kNumAgents,
            kEnvId,
            &loc_z,
            &loc_y,
            &loc_x
          );
          loc_z_arr[kAgentIdx] = loc_z;
          loc_y_arr[kAgentIdx] = loc_y;
          loc_x_arr[kAgentIdx] = loc_x;
        }

        // Reset goal point
        GenerateUnoccupiedPoint(
          kRadius,
          kSpaceLength,
          loc_z_arr,
          loc_y_arr,
          loc_x_arr,
          kNumAgents,
          kEnvId,
          &loc_z,
          &loc_y,
          &loc_x
        );
        goal_point_arr[kEnvId * DIMS    ] = loc_z;
        goal_point_arr[kEnvId * DIMS + 1] = loc_y;
        goal_point_arr[kEnvId * DIMS + 2] = loc_x;

        // Reset agent orientations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          // Use agent's state to generate a random orientation
          curandState_t* state = states[kAgentIdx];
          // Generate a random orientation from uniform distribution over [0, SYMMETRY_ORDER - 1]
          orientation_arr[kAgentIdx] = SYMMETRY_ORDER * (1.0 - curand_uniform(state));
        }
      }

      loc_z = loc_z_arr[kThisAgentArrayIdx];
      loc_y = loc_y_arr[kThisAgentArrayIdx];
      loc_x = loc_x_arr[kThisAgentArrayIdx];
      RotateCoordinates(kRadius, orientation_arr[kThisAgentArrayIdx], &loc_z, &loc_y, &loc_x);
      obs_arr[kThisAgentIdxOffset    ] = loc_z;
      obs_arr[kThisAgentIdxOffset + 1] = loc_y;
      obs_arr[kThisAgentIdxOffset + 2] = loc_x;

      loc_z = goal_point_arr[kEnvId * DIMS    ];
      loc_y = goal_point_arr[kEnvId * DIMS + 1];
      loc_x = goal_point_arr[kEnvId * DIMS + 2];
      RotateCoordinates(kRadius, orientation_arr[kThisAgentArrayIdx], &loc_z, &loc_y, &loc_x);
      obs_arr[kThisAgentIdxOffset + 3] = loc_z - obs_arr[kThisAgentIdxOffset    ];
      obs_arr[kThisAgentIdxOffset + 4] = loc_y - obs_arr[kThisAgentIdxOffset + 1];
      obs_arr[kThisAgentIdxOffset + 5] = loc_x - obs_arr[kThisAgentIdxOffset + 2];
      obs_arr[kThisAgentIdxOffset + 6] = orientation_arr[kThisAgentArrayIdx];
    }
  }

  __global__ void CudaBlessedRealmStep(
    const bool kMarred,
    int kRadius,
    int * loc_z_arr,
    int * loc_y_arr,
    int * loc_x_arr,
    int * orientation_arr,
    int * agent_types_arr,
    int * goal_point_arr,
    float * obs_arr,
    int * action_indices_arr,
    float * rewards_arr,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActions = 2;
    const int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActions + kThisAgentId * kNumActions;
    const int kSpaceLength = 2*kRadius + 1;

    int action_turn = action_indices_arr[kThisAgentActionIdxOffset    ];
    int action_move = action_indices_arr[kThisAgentActionIdxOffset + 1];

    int loc_z_tmp = loc_z_arr[kThisAgentArrayIdx];
    int loc_y_tmp = loc_y_arr[kThisAgentArrayIdx];
    int loc_x_tmp = loc_x_arr[kThisAgentArrayIdx];

    if (action_move == FORWARD) {
      if        (orientation_arr[kThisAgentArrayIdx] == NORTH    ) {
        loc_z_tmp += 1;
        loc_y_tmp -= 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == NORTHEAST) {
        loc_y_tmp -= 1;
        loc_x_tmp += 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == SOUTHEAST) {
        loc_z_tmp -= 1;
        loc_x_tmp += 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == SOUTH    ) {
        loc_z_tmp -= 1;
        loc_y_tmp += 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == SOUTHWEST) {
        loc_y_tmp += 1;
        loc_x_tmp -= 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == NORTHWEST) {
        loc_z_tmp += 1;
        loc_x_tmp -= 1;
      }
    } else if (action_turn == LEFT) {
      orientation_arr[kThisAgentArrayIdx] = (orientation_arr[kThisAgentArrayIdx] + SYMMETRY_ORDER - 1) % SYMMETRY_ORDER;
    } else if (action_turn == RIGHT) {
      orientation_arr[kThisAgentArrayIdx] = (orientation_arr[kThisAgentArrayIdx] +                  1) % SYMMETRY_ORDER;
    }

    if (
      PointIsWithinBounds(
        kRadius,
        kSpaceLength,
        loc_z_tmp,
        loc_y_tmp,
        loc_x_tmp
      )  // && !PointIsOccupied(
      //   loc_z_arr,
      //   loc_y_arr,
      //   loc_x_arr,
      //   kNumAgents,
      //   kEnvId,
      //   kThisAgentId,
      //   loc_z_tmp,
      //   loc_y_tmp,
      //   loc_x_tmp
      // )
    ) {
      // Update the point of the agent
      loc_z_arr[kThisAgentArrayIdx] = loc_z_tmp;
      loc_y_arr[kThisAgentArrayIdx] = loc_y_tmp;
      loc_x_arr[kThisAgentArrayIdx] = loc_x_tmp;
    }

    // Wait here until timestep has been updated
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }

    // Make sure all agents have updated their states
    __sync_env_threads();

    assert(0 < env_timestep_arr[kEnvId] && env_timestep_arr[kEnvId] <= kEpisodeLength);

    // -------------------------------
    // Compute reward
    // -------------------------------
    CudaBlessedRealmComputeReward(
      loc_z_arr,
      loc_y_arr,
      loc_x_arr,
      goal_point_arr,
      rewards_arr,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx
    );

    // -------------------------------
    // Generate observation
    // -------------------------------
    CudaBlessedRealmGenerateObservation(
      kRadius,
      kSpaceLength,
      loc_z_arr,
      loc_y_arr,
      loc_x_arr,
      orientation_arr,
      goal_point_arr,
      obs_arr,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx
    );
  }
}
