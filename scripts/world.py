import pygame as pg
import numpy as np
import time

from realm import Seen as Realm

PLAYER_A = 0
PLAYER_B = 1


# Create an instance of the Seen environment
world = Realm(num_agents=2)
world.reset()
len_str_episode_length = len(str(world.episode_length))

# Initial key presses
keys = pg.key.get_pressed()

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

    # Default actions
    actions = {agent_id: np.array(
        [
            world.actions[world.TURN]["NONE"],
            world.actions[world.MOVE]["NONE"],
        ], dtype=world.int_dtype
    ) for agent_id in range(world.num_agents)}

    buttons = ["", ""]

    action = False

    if keys[pg.K_UP] and (not prev_keys[pg.K_UP] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_A] and world.last_move_legal[PLAYER_A])):
        actions[PLAYER_A][world.MOVE] = world.actions[world.MOVE]["FORWARD"]
        action = True
        buttons[PLAYER_A] = "UP"
    elif keys[pg.K_RIGHT] and (not prev_keys[pg.K_RIGHT] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_A])):
        actions[PLAYER_A][world.TURN] = world.actions[world.TURN]["RIGHT"]
        action = True
        buttons[PLAYER_A] = "RIGHT"
    elif keys[pg.K_LEFT] and (not prev_keys[pg.K_LEFT] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_A])):
        actions[PLAYER_A][world.TURN] = world.actions[world.TURN]["LEFT"]
        action = True
        buttons[PLAYER_A] = "LEFT"
    
    if keys[pg.K_i] and (not prev_keys[pg.K_i] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_B] and world.last_move_legal[PLAYER_B])):
        actions[PLAYER_B][world.MOVE] = world.actions[world.MOVE]["FORWARD"]
        action = True
        buttons[PLAYER_B] = "i"
    elif keys[pg.K_l] and (not prev_keys[pg.K_l] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_B])):
        actions[PLAYER_B][world.TURN] = world.actions[world.TURN]["RIGHT"]
        action = True
        buttons[PLAYER_B] = "l"
    elif keys[pg.K_j] and (not prev_keys[pg.K_j] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_B])):
        actions[PLAYER_B][world.TURN] = world.actions[world.TURN]["LEFT"]
        action = True
        buttons[PLAYER_B] = "j"

    if action:
        t_press = time.time()
        _, rewards, done, _ = world.step(actions)
        print(f"{world.time_step - 1:>{len_str_episode_length}}:\n"
              f"  actions: {buttons}\n"
              f"  rewards: {rewards}")
        if done:
            world.reset()

    # Render the game
    world.render()

# Clean up
pg.quit()
