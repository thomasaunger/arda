__constant__ int NUM_COORDINATES = 2;

__constant__ int NUM_ORIENTATIONS = 4;

__constant__ int NORTH = 0;
__constant__ int EAST  = 1;
__constant__ int SOUTH = 2;
__constant__ int WEST  = 3;

__constant__ int LEFT  = 1;
__constant__ int RIGHT = 2;

__constant__ int FORWARD = 1;

extern "C" {
  // Device helper function to rotate coordinates
  __device__ void RotateCoordinates(
    int kGridLength,
    int * loc_y,
    int * loc_x,
    int orientation
  ) {
    int loc_y_tmp = *loc_y;
    int loc_x_tmp = *loc_x;
    if (orientation == NORTH) {
      *loc_y = loc_y_tmp;
      *loc_x = loc_x_tmp;
    } else if (orientation == EAST) {
      *loc_y = kGridLength - 1 - loc_x_tmp;
      *loc_x = loc_y_tmp;
    } else if (orientation == SOUTH) {
      *loc_y = kGridLength - 1 - loc_y_tmp;
      *loc_x = kGridLength - 1 - loc_x_tmp;
    } else if (orientation == WEST) {
      *loc_y = loc_x_tmp;
      *loc_x = kGridLength - 1 - loc_y_tmp;
    }
  }

  // Device helper function to check whether a location is within bounds
  __device__ bool LocationIsWithinBounds(
    int kGridLength,
    int loc_y,
    int loc_x
  ) {
    return (0 <= loc_y && loc_y < kGridLength && 0 <= loc_x && loc_x < kGridLength);
  }

  // Device helper function to check whether a location is occupied
  __device__ bool LocationIsOccupied(
    int * loc_y_arr,
    int * loc_x_arr,
    int kNumAgents,
    int kEnvId,
    int kThisAgentId,
    int loc_y,
    int loc_x
  ) {
    for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
      if (kAgentId != kThisAgentId) {
        int kAgentIdx = kEnvId * kNumAgents + kAgentId;
        if (loc_y_arr[kAgentIdx] == loc_y && loc_x_arr[kAgentIdx] == loc_x) {
          return true;
        }
      }
    }
    return false;
  }

  // Device helper function to generate an unoccupied location
  __device__ void GenerateUnoccupiedLocation(
    int kGridLength,
    int * loc_y_arr,
    int * loc_x_arr,
    int kNumAgents,
    int kEnvId,
    int * loc_y,
    int * loc_x
  ) {
    // Use last agent's state to generate a random location
    curandState_t* state = states[kEnvId * kNumAgents + kNumAgents - 1];
    do {
      // Generate random coordinates from uniform distribution over [0, kGridLength - 1]
      *loc_y = kGridLength * (1.0 - curand_uniform(state));
      *loc_x = kGridLength * (1.0 - curand_uniform(state));
    } while (LocationIsOccupied(
      loc_y_arr,
      loc_x_arr,
      kNumAgents,
      kEnvId,
      -1,
      *loc_y,
      *loc_x
    ));
  }

  // Device helper function to compute rewards
  __device__ void CudaBlessedRealmComputeReward(
    int * loc_y_arr,
    int * loc_x_arr,
    int * goal_location_arr,
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
      if (loc_y_arr[kThisAgentArrayIdx] == goal_location_arr[kEnvId * NUM_COORDINATES    ] &&
          loc_x_arr[kThisAgentArrayIdx] == goal_location_arr[kEnvId * NUM_COORDINATES + 1]) {
        rewards_arr[kThisAgentArrayIdx] = 1.0 * (1.0 - env_timestep_arr[kEnvId] / float(kEpisodeLength));
        // done_arr[kEnvId] = 1;
      }

      // Use only last agent's thread to check whether the maximum number of timesteps has been reached
      if (kThisAgentId == kNumAgents - 1) {
        if (env_timestep_arr[kEnvId] == kEpisodeLength - 1) {
            done_arr[kEnvId] = 1;
        }
      }
    }
  }

  // Device helper function to generate observation
  __device__ void CudaBlessedRealmGenerateObservation(
    int kGridLength,
    int * loc_y_arr,
    int * loc_x_arr,
    int * orientation_arr,
    int * goal_location_arr,
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
      // obs shape is (num_envs, kNumAgents, 1 + kGridLength * kGridLength + kNumAgents)
      const int kThisAgentIdxOffset = (kEnvId * kNumAgents + kThisAgentId) * (1 + kGridLength * kGridLength + kNumAgents);

      // Initialize obs
      for (int kLocationIdx = 0; kLocationIdx < kGridLength * kGridLength; kLocationIdx++) {
        obs_arr[kThisAgentIdxOffset + 1 + kLocationIdx] = 0.0;
      }

      int kAgentIdx;
      int loc_y;
      int loc_x;
      if (done_arr[kEnvId]) {
        // Reinitialize agent locations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          loc_y_arr[kAgentIdx] = -1;
          loc_x_arr[kAgentIdx] = -1;
        }

        // Reset agent locations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          GenerateUnoccupiedLocation(
            kGridLength,
            loc_y_arr,
            loc_x_arr,
            kNumAgents,
            kEnvId,
            &loc_y,
            &loc_x
          );
          loc_y_arr[kAgentIdx] = loc_y;
          loc_x_arr[kAgentIdx] = loc_x;
        }

        // Reset goal location
        GenerateUnoccupiedLocation(
          kGridLength,
          loc_y_arr,
          loc_x_arr,
          kNumAgents,
          kEnvId,
          &loc_y,
          &loc_x
        );
        goal_location_arr[kEnvId * NUM_COORDINATES    ] = loc_y;
        goal_location_arr[kEnvId * NUM_COORDINATES + 1] = loc_x;

        // Reset agent orientations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          // Use agent's state to generate a random orientation
          curandState_t* state = states[kAgentIdx];
          // Generate a random orientation from uniform distribution over [0, NUM_ORIENTATIONS - 1]
          orientation_arr[kAgentIdx] = NUM_ORIENTATIONS * (1.0 - curand_uniform(state));
        }
      }

      loc_y = goal_location_arr[kEnvId * NUM_COORDINATES    ];
      loc_x = goal_location_arr[kEnvId * NUM_COORDINATES + 1];
      RotateCoordinates(kGridLength, &loc_y, &loc_x, orientation_arr[kThisAgentArrayIdx]);
      obs_arr[kThisAgentIdxOffset + 1 + loc_y * kGridLength + loc_x] = 1;

      for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
        kAgentIdx = kEnvId * kNumAgents + kAgentId;

        // Add agent location
        loc_y = loc_y_arr[kAgentIdx];
        loc_x = loc_x_arr[kAgentIdx];
        RotateCoordinates(kGridLength, &loc_y, &loc_x, orientation_arr[kThisAgentArrayIdx]);
        obs_arr[kThisAgentIdxOffset + 1 + loc_y * kGridLength + loc_x] = kAgentIdx + 2;

        // Set agent orientation
        obs_arr[kThisAgentIdxOffset + 1 + kGridLength * kGridLength + kAgentId] = (orientation_arr[kAgentIdx] + NUM_ORIENTATIONS - orientation_arr[kThisAgentArrayIdx]) % NUM_ORIENTATIONS;
      }
    }
  }

  // Device helper function to generate observation
  __device__ void CudaBlessedRealmGenerateObservation2(
    int kGridLength,
    int * loc_y_arr,
    int * loc_x_arr,
    int * orientation_arr,
    int * goal_location_arr,
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
      // obs shape is (num_envs, kNumAgents, 5)
      const int kThisAgentIdxOffset = (kEnvId * kNumAgents + kThisAgentId) * 5;

      // Initialize obs
      for (int i = 0; i < 5; i++) {
        obs_arr[kThisAgentIdxOffset + i] = 0.0;
      }

      int kAgentIdx;
      int loc_y;
      int loc_x;
      if (done_arr[kEnvId]) {
        // Reinitialize agent locations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          loc_y_arr[kAgentIdx] = -1;
          loc_x_arr[kAgentIdx] = -1;
        }

        // Reset agent locations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          GenerateUnoccupiedLocation(
            kGridLength,
            loc_y_arr,
            loc_x_arr,
            kNumAgents,
            kEnvId,
            &loc_y,
            &loc_x
          );
          loc_y_arr[kAgentIdx] = loc_y;
          loc_x_arr[kAgentIdx] = loc_x;
        }

        // Reset goal location
        GenerateUnoccupiedLocation(
          kGridLength,
          loc_y_arr,
          loc_x_arr,
          kNumAgents,
          kEnvId,
          &loc_y,
          &loc_x
        );
        goal_location_arr[kEnvId * NUM_COORDINATES    ] = loc_y;
        goal_location_arr[kEnvId * NUM_COORDINATES + 1] = loc_x;

        // Reset agent orientations
        for (int kAgentId = 0; kAgentId < kNumAgents; kAgentId++) {
          kAgentIdx = kEnvId * kNumAgents + kAgentId;
          // Use agent's state to generate a random orientation
          curandState_t* state = states[kAgentIdx];
          // Generate a random orientation from uniform distribution over [0, NUM_ORIENTATIONS - 1]
          orientation_arr[kAgentIdx] = NUM_ORIENTATIONS * (1.0 - curand_uniform(state));
        }
      }

      obs_arr[kThisAgentIdxOffset    ] = loc_y_arr[kThisAgentArrayIdx];
      obs_arr[kThisAgentIdxOffset + 1] = loc_x_arr[kThisAgentArrayIdx];
      obs_arr[kThisAgentIdxOffset + 2] = orientation_arr[kThisAgentArrayIdx];
      obs_arr[kThisAgentIdxOffset + 3] = goal_location_arr[kEnvId * NUM_COORDINATES    ];
      obs_arr[kThisAgentIdxOffset + 4] = goal_location_arr[kEnvId * NUM_COORDINATES + 1];
    }
  }

  __global__ void CudaBlessedRealmStep(
    const bool kMarred,
    int kGridLength,
    int * loc_y_arr,
    int * loc_x_arr,
    int * orientation_arr,
    int * agent_types_arr,
    int * goal_location_arr,
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

    int action_turn = action_indices_arr[kThisAgentActionIdxOffset    ];
    int action_move = action_indices_arr[kThisAgentActionIdxOffset + 1];

    int loc_y_tmp = loc_y_arr[kThisAgentArrayIdx];
    int loc_x_tmp = loc_x_arr[kThisAgentArrayIdx];

    if (action_move == FORWARD) {
      if        (orientation_arr[kThisAgentArrayIdx] == NORTH) {
        loc_y_tmp -= 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == EAST ) {
        loc_x_tmp += 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == SOUTH) {
        loc_y_tmp += 1;
      } else if (orientation_arr[kThisAgentArrayIdx] == WEST ) {
        loc_x_tmp -= 1;
      }
    } else if (action_turn == LEFT) {
      orientation_arr[kThisAgentArrayIdx] = (orientation_arr[kThisAgentArrayIdx] + NUM_ORIENTATIONS - 1) % NUM_ORIENTATIONS;
    } else if (action_turn == RIGHT) {
      orientation_arr[kThisAgentArrayIdx] = (orientation_arr[kThisAgentArrayIdx] +                    1) % NUM_ORIENTATIONS;
    }

    if (
      LocationIsWithinBounds(
        kGridLength,
        loc_y_tmp,
        loc_x_tmp
      ) && !LocationIsOccupied(
        loc_y_arr,
        loc_x_arr,
        kNumAgents,
        kEnvId,
        kThisAgentId,
        loc_y_tmp,
        loc_x_tmp
      )
    ) {
      // Update the location of the agent
      loc_y_arr[kThisAgentArrayIdx] = loc_y_tmp;
      loc_x_arr[kThisAgentArrayIdx] = loc_x_tmp;
    }

    // assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <= kEpisodeLength);

    // Make sure all agents have updated their states
    __sync_env_threads();

    // -------------------------------
    // Compute reward
    // -------------------------------
    CudaBlessedRealmComputeReward(
      loc_y_arr,
      loc_x_arr,
      goal_location_arr,
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
    CudaBlessedRealmGenerateObservation2(
      kGridLength,
      loc_y_arr,
      loc_x_arr,
      orientation_arr,
      goal_location_arr,
      obs_arr,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx
    );

    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == kNumAgents - 1) {
      env_timestep_arr[kEnvId] += 1;
    }
  }
}
