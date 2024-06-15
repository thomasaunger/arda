import numpy as np
# import os
import pygame as pg
import time
# import torch

from realm import HexagonalSeen as Realm

# from torch.distributions.categorical import Categorical

# from warp_drive.training.models.fully_connected         import FullyConnected
from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
    recursive_obs_dict_to_spaces_dict,
)

NUM_AGENTS = 2

PLAYER_A = 0
PLAYER_B = 1

SAMPLE = False


# Create an instance of the Seen environment
world = Realm(num_agents=NUM_AGENTS)
obs = world.reset()
world.observation_space = recursive_obs_dict_to_spaces_dict(obs)
len_str_episode_length = len(str(world.episode_length))

pg.key.set_repeat(150, 150) # delay, interval in ms

key_map = {
    pg.K_UP:    (PLAYER_A, world.space.action_space.MOVE.FORWARD),
    pg.K_LEFT:  (PLAYER_A, world.space.action_space.TURN.LEFT),
    pg.K_RIGHT: (PLAYER_A, world.space.action_space.TURN.RIGHT),

    pg.K_i:     (PLAYER_B, world.space.action_space.MOVE.FORWARD),
    pg.K_j:     (PLAYER_B, world.space.action_space.TURN.LEFT),
    pg.K_l:     (PLAYER_B, world.space.action_space.TURN.RIGHT),
}

# # Summon spirit
# spirit_config = dict(
#     type="fully_connected", fc_dims=[1024, 1024, 1024], spirit_ckpt_filepath=""
# )
# policy = "power"
# policy_tag_to_agent_id_map = {
#     "power": world.powers,
#     "angel": world.angels,
# }
# spirit = FullyConnected(
#     world,
#     spirit_config,
#     policy,
#     policy_tag_to_agent_id_map
# )
# spirit.load_state_dict(
#     torch.load(
#         os.path.join(
#             "spirits",
#             "power_25591808.state_dict"
#         ),
#         map_location=torch.device("cpu")
#     )
# )
# spirit.eval()

total_rewards  = np.zeros(2, dtype=world.float_dtype)
total_episodes = np.ones( 1, dtype=world.int_dtype)

# Game loop
t_action = np.array([time.time()] * NUM_AGENTS)
t_delta = 0.15
running = True
while running:
    # Default actions
    actions = {
        agent_id: np.array(
            [
                world.space.action_space.TURN.NONE,
                world.space.action_space.MOVE.NONE,
            ], dtype=world.int_dtype
        ) for agent_id in range(world.num_agents)
    }

    # Spirit actions
    # agent_p = world.powers[0]
    # if t_delta < time.time() - t_action[agent_p]:
    #     dists = [Categorical(probs=probs) for probs in spirit(torch.from_numpy(obs[agent_p]))[0]]
    #     actions[agent_p] = [dist.sample() for dist in dists] if SAMPLE else [torch.argmax(dist.probs) for dist in dists]
    #     t_action[agent_p] = time.time()

    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            # Human actions
            try:
                agent, action = key_map[event.key]
                actions[agent][action.type] = action
                t_action[agent] = time.time()
            except KeyError:
                pass

    if any([np.any(actions_i) for actions_i in actions.values()]):
        obs, rewards, done, _ = world.step(actions)
        total_rewards += rewards
        print(f"{world.time_step - 1:>{len_str_episode_length}}:\n"
              f"  rewards: {rewards}\n"
              f"  running average rewards:"
        )
        for total_reward in total_rewards:
              print(f"    {total_reward/total_episodes[0]} = {total_reward}/{total_episodes[0]}")
        if done:
            world.reset()
            total_episodes += 1

    # Render the game
    world.render()

# Clean up
pg.quit()
