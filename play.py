import pygame as pg
import numpy as np

from arda import Arda

PLAYER = 0


# Initialize pygame
pg.init()

# Set up the game window
screen_width = 1920
screen_height = 1080
screen = pg.display.set_mode((screen_width, screen_height))
pg.display.set_caption("Arda")

# Create an instance of the Arda environment
env = Arda(screen)
env.reset()

# initial key presses
keys = pg.key.get_pressed()

timestep = 0

# Game loop
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
            Arda.actions[Arda.RED]["NONE"],
            Arda.actions[Arda.GREEN]["NONE"],
            Arda.actions[Arda.BLUE]["NONE"],
        ], dtype=np.int8
    ) for agent_id in range(env.num_agents)}

    if keys[pg.K_UP]:
        print("UP")
        actions[PLAYER][Arda.RED] = Arda.actions[Arda.RED]["MORE"]
    elif keys[pg.K_DOWN]:
        print("DOWN")
        actions[PLAYER][Arda.RED] = Arda.actions[Arda.RED]["LESS"]
    
    if keys[pg.K_RIGHT]:
        print("RIGHT")
        actions[PLAYER][Arda.GREEN] = Arda.actions[Arda.GREEN]["MORE"]
    elif keys[pg.K_LEFT]:
        print("LEFT")
        actions[PLAYER][Arda.GREEN] = Arda.actions[Arda.GREEN]["LESS"]
    
    if keys[pg.K_SPACE]:
        print("SPACE")
        actions[PLAYER][Arda.BLUE] = Arda.actions[Arda.BLUE]["MORE"]
    
    _, _, done, _ = env.step(actions)
    
    if done:
        env.reset()

    # Render the game
    env.render()

    timestep += 1

# Clean up
pg.quit()
