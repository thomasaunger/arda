import pygame as pg
import numpy as np
import time

from seen import Seen as Realm

PLAYER_A = 0
PLAYER_B = 1


# Create an instance of the Seen environment
world = Realm(num_agents=2)
world.reset()
first_press = [True]*world.num_agents

# initial key presses
keys = pg.key.get_pressed()

timestep = 0

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

    change = False

    t_delta = 0.1

    if keys[pg.K_UP] and (not prev_keys[pg.K_UP] or (time.time() - t_press > t_delta and not first_press[PLAYER_A])):
        print("UP")
        actions[PLAYER_A][Realm.MOVE] = Realm.actions[Realm.MOVE]["FORWARD"]
        change = True
        first_press[PLAYER_A] = False
    elif keys[pg.K_RIGHT] and (not prev_keys[pg.K_RIGHT] or (time.time() - t_press > t_delta and not first_press[PLAYER_A])):
        print("RIGHT")
        actions[PLAYER_A][Realm.TURN] = Realm.actions[Realm.TURN]["RIGHT"]
        change = True
        first_press[PLAYER_A] = False
    elif keys[pg.K_LEFT] and (not prev_keys[pg.K_LEFT] or (time.time() - t_press > t_delta and not first_press[PLAYER_A])):
        print("LEFT")
        actions[PLAYER_A][Realm.TURN] = Realm.actions[Realm.TURN]["LEFT"]
        change = True
        first_press[PLAYER_A] = False
    
    if keys[pg.K_i] and (not prev_keys[pg.K_i] or (time.time() - t_press > t_delta and not first_press[PLAYER_B])):
        print("i")
        actions[PLAYER_B][Realm.MOVE] = Realm.actions[Realm.MOVE]["FORWARD"]
        change = True
        first_press[PLAYER_B] = False
    elif keys[pg.K_l] and (not prev_keys[pg.K_l] or (time.time() - t_press > t_delta and not first_press[PLAYER_B])):
        print("l")
        actions[PLAYER_B][Realm.TURN] = Realm.actions[Realm.TURN]["RIGHT"]
        change = True
        first_press[PLAYER_B] = False
    elif keys[pg.K_j] and (not prev_keys[pg.K_j] or (time.time() - t_press > t_delta and not first_press[PLAYER_B])):
        print("j")
        actions[PLAYER_B][Realm.TURN] = Realm.actions[Realm.TURN]["LEFT"]
        change = True
        first_press[PLAYER_B] = False

    if change:
        t_press = time.time()
        _, _, done, _ = world.step(actions)
        if done:
            world.reset()
            first_press = [True]*world.num_agents

    # Render the game
    world.render()

    timestep += 1

# Clean up
pg.quit()
