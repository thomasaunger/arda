import numpy as np
import os
import pygame as pg
import time
import torch

from realm import Seen as Realm

from torch.distributions.categorical import Categorical

from warp_drive.training.models.fully_connected         import FullyConnected
from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
    recursive_obs_dict_to_spaces_dict,
)

PLAYER_A = 0
PLAYER_B = 1

HALTING = False


# Create an instance of the Seen environment
world = Realm(num_agents=1, grid_length=4)
obs = world.reset()
world.observation_space = recursive_obs_dict_to_spaces_dict(obs)
len_str_episode_length = len(str(world.episode_length))

# Summon spirit
spirit_config = dict(
    type="fully_connected", fc_dims=[1024, 1024, 1024], spirit_ckpt_filepath=""
)
policy = "power"
policy_tag_to_agent_id_map = {
    "power": list(world.powers),
    "angel": list(world.angels),
}
spirit = FullyConnected(
    world,
    spirit_config,
    policy,
    policy_tag_to_agent_id_map
)
spirit.load_state_dict(
    torch.load(
        os.path.join(
            "spirits",
            # "angel_6399488.state_dict"
            # "power_3199488.state_dict"
            # "power_6389760.state_dict"
            # "angel_6389760.state_dict"
            # "power_40960000.state_dict"
            # "angel_40960000.state_dict"
            "power_76791808.state_dict"
        ),
        map_location=torch.device("cpu")
    )
)
spirit.eval()

# Initial key presses
keys = pg.key.get_pressed()

spirit_actions = [torch.tensor(0), torch.tensor(0)]

total_rewards  = np.zeros(2, dtype=world.float_dtype)
total_episodes = np.ones( 1, dtype=world.int_dtype)

# Game loop
t_press = 1000000
t_delta = 0.15
running = True
while running:
    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    prev_keys = keys
    keys = pg.key.get_pressed()

    out = spirit(torch.from_numpy(obs[list(world.powers)[0]]))
    dists = [Categorical(probs=probs) for probs in out[0]]
    prev_spirit_actions = spirit_actions
    spirit_actions = [dist.sample() for dist in dists]

    # Default actions
    actions = {agent_id: np.array(
        [
            world.actions[world.TURN]["NONE"],
            world.actions[world.MOVE]["NONE"],
        ], dtype=world.int_dtype
    ) for agent_id in range(world.num_agents)}

    buttons = ["", ""]

    action = False

    if (keys[pg.K_UP] or (PLAYER_A == list(world.powers)[0] and spirit_actions[1].item() == 1)) and (not (prev_keys[pg.K_UP] or (PLAYER_A == list(world.powers)[0] and prev_spirit_actions[1].item() == 1)) or (time.time() - t_press > t_delta and (not world.first_action[PLAYER_A] or not HALTING) and world.last_move_legal[PLAYER_A])):
        actions[PLAYER_A][world.MOVE] = world.actions[world.MOVE]["FORWARD"]
        action = True
        buttons[PLAYER_A] = "UP"
    elif (keys[pg.K_LEFT] or (PLAYER_A == list(world.powers)[0] and spirit_actions[0].item() == 1)) and (not (prev_keys[pg.K_LEFT] or (PLAYER_A == list(world.powers)[0] and prev_spirit_actions[0].item() == 1)) or (time.time() - t_press > t_delta and (not world.first_action[PLAYER_A] or not HALTING))):
        actions[PLAYER_A][world.TURN] = world.actions[world.TURN]["LEFT"]
        action = True
        buttons[PLAYER_A] = "LEFT"
    elif (keys[pg.K_RIGHT] or (PLAYER_A == list(world.powers)[0] and spirit_actions[0].item() == 2)) and (not (prev_keys[pg.K_RIGHT] or (PLAYER_A == list(world.powers)[0] and prev_spirit_actions[0].item() == 2)) or (time.time() - t_press > t_delta and (not world.first_action[PLAYER_A] or not HALTING))):
        actions[PLAYER_A][world.TURN] = world.actions[world.TURN]["RIGHT"]
        action = True
        buttons[PLAYER_A] = "RIGHT"
    
    if (keys[pg.K_i] or (PLAYER_B == list(world.powers)[0] and spirit_actions[1].item() == 1)) and (not (prev_keys[pg.K_i] or (PLAYER_B == list(world.powers)[0] and prev_spirit_actions[1].item() == 1)) or (time.time() - t_press > t_delta and (not world.first_action[PLAYER_B] or not HALTING) and world.last_move_legal[PLAYER_B])):
        actions[PLAYER_B][world.MOVE] = world.actions[world.MOVE]["FORWARD"]
        action = True
        buttons[PLAYER_B] = "i"
    elif (keys[pg.K_j] or (PLAYER_B == list(world.powers)[0] and spirit_actions[0].item() == 1)) and (not (prev_keys[pg.K_j] or (PLAYER_B == list(world.powers)[0] and prev_spirit_actions[0].item() == 1)) or (time.time() - t_press > t_delta and (not world.first_action[PLAYER_B] or not HALTING))):
        actions[PLAYER_B][world.TURN] = world.actions[world.TURN]["LEFT"]
        action = True
        buttons[PLAYER_B] = "j"
    elif (keys[pg.K_l] or (PLAYER_B == list(world.powers)[0] and spirit_actions[0].item() == 2)) and (not (prev_keys[pg.K_l] or (PLAYER_B == list(world.powers)[0] and prev_spirit_actions[0].item() == 2)) or (time.time() - t_press > t_delta and (not world.first_action[PLAYER_B] or not HALTING))):
        actions[PLAYER_B][world.TURN] = world.actions[world.TURN]["RIGHT"]
        action = True
        buttons[PLAYER_B] = "l"

    if action:
        t_press = time.time()
        obs, rewards, done, _ = world.step(actions)
        total_rewards += rewards
        print(f"{world.time_step - 1:>{len_str_episode_length}}:\n"
              f"  actions: {buttons}\n"
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
