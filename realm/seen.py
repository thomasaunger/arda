import pygame as pg

from .blessed_realm import BlessedRealm as Realm

SCREEN_LENGTH_Y = 480
SCREEN_LENGTH_X = 640

CELL_LENGTH_Y = 32
CELL_LENGTH_X = 32

COORDINATE_Y = 0
COORDINATE_X = 1

MARGIN = 2


class Seen(Realm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize pygame
        pg.init()

        # Create a window
        self.screen = pg.display.set_mode((SCREEN_LENGTH_X, SCREEN_LENGTH_Y))
        
        pg.display.set_caption("Realm")

    def _agent_image(self, agent_id):
        y_offset = (self.screen.get_height() - self.space.array.shape[COORDINATE_Y]*CELL_LENGTH_Y)//2
        x_offset = (self.screen.get_width()  - self.space.array.shape[COORDINATE_X]*CELL_LENGTH_X)//2

        y = self.space.agent_points[agent_id][COORDINATE_Y]
        x = self.space.agent_points[agent_id][COORDINATE_X]

        match self.space.agent_orientations[agent_id]:
            case self.space.NORTH:
                return [
                    (x_offset +  x     *CELL_LENGTH_X + MARGIN,           y_offset + (y + 1)*CELL_LENGTH_Y - MARGIN),
                    (x_offset + (x + 1)*CELL_LENGTH_X - MARGIN,           y_offset + (y + 1)*CELL_LENGTH_Y - MARGIN),
                    (x_offset +  x     *CELL_LENGTH_X + CELL_LENGTH_X//2, y_offset +  y     *CELL_LENGTH_Y + MARGIN),
                ]
            case self.space.EAST:
                return [
                    (x_offset +  x     *CELL_LENGTH_X + MARGIN, y_offset +  y     *CELL_LENGTH_Y + MARGIN),
                    (x_offset +  x     *CELL_LENGTH_X + MARGIN, y_offset + (y + 1)*CELL_LENGTH_Y - MARGIN),
                    (x_offset + (x + 1)*CELL_LENGTH_X - MARGIN, y_offset +  y     *CELL_LENGTH_Y + CELL_LENGTH_Y//2),
                ]
            case self.space.SOUTH:
                return [
                    (x_offset + (x + 1)*CELL_LENGTH_X - MARGIN,           y_offset +  y     *CELL_LENGTH_Y + MARGIN),
                    (x_offset +  x     *CELL_LENGTH_X + MARGIN,           y_offset +  y     *CELL_LENGTH_Y + MARGIN),
                    (x_offset +  x     *CELL_LENGTH_X + CELL_LENGTH_X//2, y_offset + (y + 1)*CELL_LENGTH_Y - MARGIN),
                ]
            case self.space.WEST:
                return [
                    (x_offset + (x + 1)*CELL_LENGTH_X - MARGIN, y_offset + (y + 1)*CELL_LENGTH_Y - MARGIN),
                    (x_offset + (x + 1)*CELL_LENGTH_X - MARGIN, y_offset +  y     *CELL_LENGTH_Y + MARGIN),
                    (x_offset +  x     *CELL_LENGTH_X + MARGIN, y_offset +  y     *CELL_LENGTH_Y + CELL_LENGTH_Y//2),
                ]

    def render(self, mode="men"):
        # Render the environment
        self.screen.fill((0, 0, 0))  # Fill the screen with black color

        y_offset = (self.screen.get_height() - self.space.array.shape[COORDINATE_Y]*CELL_LENGTH_Y)//2
        x_offset = (self.screen.get_width()  - self.space.array.shape[COORDINATE_X]*CELL_LENGTH_X)//2

        # Draw the agents
        for agent_id in range(self.num_agents):
            if self.space.agent_types[agent_id] == self.space.ANGEL:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            y = y_offset + self.space.agent_points[agent_id][COORDINATE_Y]*CELL_LENGTH_Y
            x = x_offset + self.space.agent_points[agent_id][COORDINATE_X]*CELL_LENGTH_X
            pg.draw.polygon(
                self.screen,
                color,
                self._agent_image(agent_id),
            )
        
        # Draw the goal as a square
        y = y_offset + self.goal_point[COORDINATE_Y]*CELL_LENGTH_Y + CELL_LENGTH_Y//2
        x = x_offset + self.goal_point[COORDINATE_X]*CELL_LENGTH_X + CELL_LENGTH_X//2
        pg.draw.rect(
            self.screen,
            (0, 255, 0),
            pg.Rect(
                x_offset + self.goal_point[COORDINATE_X]*CELL_LENGTH_X + MARGIN,
                y_offset + self.goal_point[COORDINATE_Y]*CELL_LENGTH_Y + MARGIN,
                CELL_LENGTH_X - 2*MARGIN + 1,
                CELL_LENGTH_Y - 2*MARGIN + 1,
            ),
        )

        # Draw the space
        for m in range(self.space.array.shape[COORDINATE_Y] + 1):
            y = y_offset + m*CELL_LENGTH_Y
            pg.draw.line(self.screen, (255, 255, 255), (x_offset, y), (x_offset + self.space.array.shape[COORDINATE_Y]*CELL_LENGTH_X, y))

        for n in range(self.space.array.shape[COORDINATE_X] + 1):
            x = x_offset + n*CELL_LENGTH_X
            pg.draw.line(self.screen, (255, 255, 255), (x, y_offset), (x, y_offset + self.space.array.shape[COORDINATE_X]*CELL_LENGTH_Y))

        pg.display.flip()  # Update the display
