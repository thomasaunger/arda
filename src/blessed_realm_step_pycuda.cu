__constant__ float kPi = 3.141592654;
__constant__ float kTwoPi = 6.283185308;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases

extern "C" {
  // Device helper function to generate observation
  __device__ void CudaBlessedRealmGenerateObservation(
    float * obs_arr,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    int num_features = 7;

    if (kThisAgentId < kNumAgents) {
      // obs shape is (num_envs, kNumAgents,
      // num_features * (kNumAgents - 1) + 1)
      const int kThisAgentIdxOffset = kEnvId * kNumAgents *
        (num_features * (kNumAgents - 1) + 1) +
        kThisAgentId * (num_features * (kNumAgents - 1) + 1);
      // Initialize obs
      int index = 0;
      for (int other_agent_id = 0; other_agent_id < kNumAgents;
      other_agent_id++) {
        if (other_agent_id != kThisAgentId) {
          obs_arr[kThisAgentIdxOffset + 0 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 1 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 2 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 3 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 4 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 5 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 6 * (kNumAgents - 1) + index]
            = 0.0;
          index += 1;
        }
      }
      obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1)] = 0.0;
    }
  }

  // Device helper function to compute rewards
  __device__ void CudaBlessedRealmComputeReward(
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
      // initialize rewards
      rewards_arr[kThisAgentArrayIdx] = 0.0; 

      // Wait here to update the number of runners before determining done_arr
      __sync_env_threads();
      // Use only agent 0's thread to set done_arr
      if (kThisAgentId == 0) {
        if (env_timestep_arr[kEnvId] == kEpisodeLength) {
            done_arr[kEnvId] = 1;
        }
      }
    }
  }

  __global__ void CudaBlessedRealmStep(
    const bool kMarred,
    int kGridLength,
    int * loc_x_arr,
    int * loc_y_arr,
    int * orientation_arr,
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

    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }

    // Generate a random float between 0 and 1.
    curandState_t* state = states[kThisAgentArrayIdx];  // Retrieve the state for this thread/agent.
    float randomValue = curand_uniform(state);  // Generate the random float.

    // Wait here until timestep has been updated
    __sync_env_threads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <=
      kEpisodeLength);

    // Make sure all agents have updated their states
    __sync_env_threads();
    // -------------------------------
    // Generate observation
    // -------------------------------
    CudaBlessedRealmGenerateObservation(
      obs_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);

    // -------------------------------
    // Compute reward
    // -------------------------------
    CudaBlessedRealmComputeReward(
      rewards_arr,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);
  }
}
