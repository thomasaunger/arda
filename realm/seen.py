import pygame as pg

from .blessed_realm import BlessedRealm as Realm

SCREEN_WIDTH  = 640
SCREEN_HEIGHT = 480


class Seen(Realm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize pygame
        pg.init()

        # Create a window
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        pg.display.set_caption("Realm")

    def _agent_image(self, i, margin):
        y_width = 32
        x_width = 32

        y_offset = (self.screen.get_height() - self.grid.shape[Realm.Y]*y_width) // 2
        x_offset = (self.screen.get_width()  - self.grid.shape[Realm.X]*x_width) // 2

        y = self.agent_locations[i][Realm.Y]
        x = self.agent_locations[i][Realm.X]

        match self.agent_orientations[i]:
            case Realm.NORTH:
                return [
                    (x_offset +  x     *x_width + margin,       y_offset + (y + 1)*y_width - margin),
                    (x_offset + (x + 1)*x_width - margin,       y_offset + (y + 1)*y_width - margin),
                    (x_offset +  x     *x_width + x_width // 2, y_offset +  y     *y_width + margin),
                ]
            case Realm.EAST:
                return [
                    (x_offset +  x     *x_width + margin, y_offset +  y     *y_width + margin),
                    (x_offset +  x     *x_width + margin, y_offset + (y + 1)*y_width - margin),
                    (x_offset + (x + 1)*x_width - margin, y_offset +  y     *y_width + y_width // 2),
                ]
            case Realm.SOUTH:
                return [
                    (x_offset + (x + 1)*x_width - margin,       y_offset +  y     *y_width + margin),
                    (x_offset +  x     *x_width + margin,       y_offset +  y     *y_width + margin),
                    (x_offset +  x     *x_width + x_width // 2, y_offset + (y + 1)*y_width - margin),
                ]
            case Realm.WEST:
                return [
                    (x_offset + (x + 1)*x_width - margin, y_offset + (y + 1)*y_width - margin),
                    (x_offset + (x + 1)*x_width - margin, y_offset +  y     *y_width + margin),
                    (x_offset +  x     *x_width + margin, y_offset + (y + 1)*y_width - y_width // 2),
                ]

    def render(self, mode="men"):
        # Render the environment
        self.screen.fill((0, 0, 0))  # Fill the screen with black color

        margin = 2

        y_width = 32
        x_width = 32

        y_offset = (self.screen.get_height() - self.grid.shape[Realm.Y]*y_width) // 2
        x_offset = (self.screen.get_width()  - self.grid.shape[Realm.X]*x_width) // 2

        # Draw the agents
        for i in range(self.num_agents):
            if i == 0:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            y = y_offset + self.agent_locations[i][Realm.Y]*y_width
            x = x_offset + self.agent_locations[i][Realm.X]*x_width
            pg.draw.polygon(
                self.screen,
                color,
                self._agent_image(i, margin),
            )
        
        # Draw the goal as a square
        y = y_offset + self.goal_location[Realm.Y]*y_width + y_width // 2
        x = x_offset + self.goal_location[Realm.X]*x_width + x_width // 2
        pg.draw.rect(
            self.screen,
            (0, 255, 0),
            pg.Rect(
                x_offset + self.goal_location[Realm.X]*x_width + margin,
                y_offset + self.goal_location[Realm.Y]*y_width + margin,
                x_width - 2*margin + 1,
                y_width - 2*margin + 1,
            ),
        )

        # Draw the grid
        for m in range(self.grid.shape[Realm.Y] + 1):
            y = y_offset + m*y_width
            pg.draw.line(self.screen, (255, 255, 255), (x_offset, y), (x_offset + self.grid.shape[Realm.Y]*x_width, y))

        for n in range(self.grid.shape[Realm.X] + 1):
            x = x_offset + n*x_width
            pg.draw.line(self.screen, (255, 255, 255), (x, y_offset), (x, y_offset + self.grid.shape[Realm.X]*y_width))

        pg.display.flip()  # Update the display
