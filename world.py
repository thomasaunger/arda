import pygame as pg
import numpy as np
import time

from seen import Seen as Realm

PLAYER_A = 0
PLAYER_B = 1


# Create an instance of the Seen environment
world = Realm(num_agents=2)
world.reset()

# initial key presses
keys = pg.key.get_pressed()

# Game loop
t_press = 1000000
running = True
while running:
    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    prev_keys = keys
    keys = pg.key.get_pressed()

    # default actions
    actions = {agent_id: np.array(
        [
            Realm.actions[Realm.TURN]["NONE"],
            Realm.actions[Realm.MOVE]["NONE"],
        ], dtype=np.int8
    ) for agent_id in range(world.num_agents)}

    action = False

    t_delta = 0.1

    if keys[pg.K_UP] and (not prev_keys[pg.K_UP] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_A] and world.last_move_legal[PLAYER_A])):
        print("UP")
        actions[PLAYER_A][Realm.MOVE] = Realm.actions[Realm.MOVE]["FORWARD"]
        action = True
    elif keys[pg.K_RIGHT] and (not prev_keys[pg.K_RIGHT] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_A])):
        print("RIGHT")
        actions[PLAYER_A][Realm.TURN] = Realm.actions[Realm.TURN]["RIGHT"]
        action = True
    elif keys[pg.K_LEFT] and (not prev_keys[pg.K_LEFT] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_A])):
        print("LEFT")
        actions[PLAYER_A][Realm.TURN] = Realm.actions[Realm.TURN]["LEFT"]
        action = True
    
    if keys[pg.K_i] and (not prev_keys[pg.K_i] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_B] and world.last_move_legal[PLAYER_B])):
        print("i")
        actions[PLAYER_B][Realm.MOVE] = Realm.actions[Realm.MOVE]["FORWARD"]
        action = True
    elif keys[pg.K_l] and (not prev_keys[pg.K_l] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_B])):
        print("l")
        actions[PLAYER_B][Realm.TURN] = Realm.actions[Realm.TURN]["RIGHT"]
        action = True
    elif keys[pg.K_j] and (not prev_keys[pg.K_j] or (time.time() - t_press > t_delta and not world.first_action[PLAYER_B])):
        print("j")
        actions[PLAYER_B][Realm.TURN] = Realm.actions[Realm.TURN]["LEFT"]
        action = True

    if action:
        t_press = time.time()
        _, _, done, _ = world.step(actions)
        if done:
            world.reset()

    # Render the game
    world.render()

# Clean up
pg.quit()
