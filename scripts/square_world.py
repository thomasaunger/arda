import numpy as np
import os
import pygame as pg
import time
import torch

from realm import SquareSeen as Realm

from torch.distributions.categorical import Categorical

from warp_drive.training.models.fully_connected         import FullyConnected
from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
    recursive_obs_dict_to_spaces_dict,
)

NUM_AGENTS = 2

PLAYER_A = 0
PLAYER_B = 1

SAMPLE = True


# Create an instance of the Seen environment
world = Realm(radius=1, num_agents=NUM_AGENTS)
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

# Summon spirit
spirit_config = dict(
    type="fully_connected", fc_dims=[1024, 1024, 1024], spirit_ckpt_filepath=""
)
policy_tag_to_agent_id_map = {
    "power": world.powers,
    "angel": world.angels,
}
power = FullyConnected(
    world,
    spirit_config,
    "power",
    policy_tag_to_agent_id_map
)
power.load_state_dict(
    torch.load(
        os.path.join(
            "spirits",
            "square",
            "speaking",
            "7",
            "power_25591808.state_dict"
        ),
        map_location=torch.device("cpu")
    )
)
power.eval()
angel = FullyConnected(
    world,
    spirit_config,
    "angel",
    policy_tag_to_agent_id_map
)
angel.load_state_dict(
    torch.load(
        os.path.join(
            "spirits",
            "square",
            "speaking",
            "7",
            "angel_25591808.state_dict"
        ),
        map_location=torch.device("cpu")
    )
)
angel.eval()

agent_p = world.powers[0]
agent_a = world.angels[0]

total_rewards  = np.zeros(2, dtype=world.float_dtype)
total_episodes = np.ones( 1, dtype=world.int_dtype)

actions = {
    agent_id: np.array(
        [
            world.space.action_space.TURN.NONE,
            world.space.action_space.MOVE.NONE,
            *[0]*world.NUM_POSITIONS,
        ], dtype=world.int_dtype
    ) for agent_id in range(world.num_agents)
}

# Game loop
t_action = np.array([time.time()] * NUM_AGENTS)
t_delta = 0.15
quit = False
while not quit:
    # Default actions
    for agent_id in range(world.num_agents):
        actions[agent_id][world.space.action_space.TURN] = world.space.action_space.TURN.NONE
        actions[agent_id][world.space.action_space.MOVE] = world.space.action_space.MOVE.NONE

    # Spirit actions
    if t_delta < time.time() - t_action[agent_a]:
        dists = [Categorical(probs=probs) for probs in angel(torch.from_numpy(
            np.concatenate(
                [
                    np.zeros(len(obs[agent_a][:-world.NUM_POSITIONS]), dtype=world.float_dtype),
                    actions[agent_p][-world.NUM_POSITIONS:].astype(world.float_dtype),
                ],
                dtype=world.float_dtype
            )
        ))[0]]
        actions[agent_a] = np.array([dist.sample().item() if SAMPLE else torch.argmax(dist.probs).item() for dist in dists], dtype=world.int_dtype)
        t_action[agent_a] = time.time()

    if t_delta < time.time() - t_action[agent_p]:
        dists = [Categorical(probs=probs) for probs in power(torch.from_numpy(
            np.concatenate(
                [
                    obs[agent_a][:-world.NUM_POSITIONS],
                    np.zeros(world.NUM_POSITIONS, dtype=world.float_dtype),
                ],
                dtype=world.float_dtype
            )
        ))[0]]
        actions[agent_p] = np.array([dist.sample().item() if SAMPLE else torch.argmax(dist.probs).item() for dist in dists], dtype=world.int_dtype)
        actions[agent_p][world.space.action_space.TURN] = world.space.action_space.TURN.NONE
        actions[agent_p][world.space.action_space.MOVE] = world.space.action_space.MOVE.NONE
        t_action[agent_p] = time.time()

    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit = True
        elif event.type == pg.KEYDOWN:
            # Human actions
            try:
                agent, action = key_map[event.key]
                actions[agent][action.type] = action
                t_action[agent] = time.time()
            except KeyError:
                pass

    if any([np.any(actions_i[:-world.NUM_POSITIONS]) for actions_i in actions.values()]):
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
