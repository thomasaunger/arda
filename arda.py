import gym
import pygame as pg


class Arda(gym.Env):

    RED   = 0
    GREEN = 1
    BLUE  = 2

    actions = {
        RED: {
            "NONE": 0,
            "MORE": 1,
            "LESS": 2,
        },
        GREEN: {
            "NONE": 0,
            "MORE": 1,
            "LESS": 2,
        },
        BLUE: {
            "NONE": 0,
            "MORE": 1,
            "LESS": 2,
        },
    }

    def __init__(self, screen, num_agents=1):
        # Define your environment variables here

        self.screen = screen

        self.num_agents = num_agents

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: gym.spaces.MultiDiscrete(
                (3, 3, 3)
            )
            for agent_id in range(self.num_agents)
        }

    def reset(self):
        # Reset the environment to its initial state
        self.colors = [0, 0, 0]

    def step(self, actions=None):
        # Perform one step in the environment

        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        for agent_id, action in actions.items():
            for color, subaction in enumerate(action):
                match subaction:
                    case 1:
                        self.colors[color] = (self.colors[color] + 1) % 256
                    case 2:
                        self.colors[color] = (self.colors[color] - 1) % 256

        obss = None
        rewards = None
        done = False
        info = None

        return obss, rewards, done, info

    def render(self, mode="human"):
        # Render the environment
        self.screen.fill(self.colors)  # Fill the screen with black color

        # Draw game objects using pygame's drawing functions
        # For example:
        # pg.draw.rect(screen, (255, 0, 0), (100, 100, 50, 50))  # Draw a red rectangle
        # pg.draw.circle(screen, (0, 255, 0), (200, 200), 30)  # Draw a green circle

        pg.display.flip()  # Update the display

    def close(self):
        # Clean up resources
        pass

# Register the environment with OpenAI Gym
gym.register(
    id="Arda-v0",
    entry_point="arda:Arda",
)
