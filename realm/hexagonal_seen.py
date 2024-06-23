import math
import numpy as np
import pygame as pg

from .utils import Seen

COORDINATE_Y = 1
COORDINATE_X = 2


class HexagonalSeen(Seen):

    def render_hexagon(self, y, x, ka, kb, m_h, color=(255, 255, 255)):
        pg.draw.polygon(
            self.screen,
            color,
            [
                (x - (kb - ka)//2      -   m_h, y + ka + m_h),
                (x - (kb - ka)//2 - ka - 2*m_h, y),
                (x - (kb - ka)//2      -   m_h, y - ka - m_h),
                (x + (kb - ka)//2      +   m_h, y - ka - m_h),
                (x + (kb - ka)//2 + ka + 2*m_h, y),
                (x + (kb - ka)//2      +   m_h, y + ka + m_h),
            ],
            1
        )
    
    def render_agent(self, y, x, ka, kb, a, b, orientation, color=(255, 0, 0)):
        m_a = 0    # margin      in kc
        h   = 1    # head height in kc
        w   = ka/2 # width       in pixels
        r   = ka/4 # head barb   in pixels

        arrow_straight = np.array(
            [
                [- ((kb - ka)//4),        + ka],
                [- ((kb - ka)//4),        - ka + h*ka],
                [- ((kb - ka)//4) - 2*r,  - ka + h*ka],
                [  0,                     - ka],
                [  ((kb - ka)//4) + 2*r,  - ka + h*ka],
                [  ((kb - ka)//4),        - ka + h*ka],
                [  ((kb - ka)//4),        + ka],
            ]
        )

        arrow_diagonal = np.array(
            [
                [- kb/2 + (m_a+h)*b -  w/2,      - ka/2 + (m_a+h)*a +  w/2     ],
                [- kb/2 + (m_a+h)*b - (w/2 + r), - ka/2 + (m_a+h)*a + (w/2 + r)],
                [- kb/2 +  m_a   *b,             - ka/2 +  m_a   *a            ],
                [- kb/2 + (m_a+h)*b + (w/2 + r), - ka/2 + (m_a+h)*a - (w/2 + r)],
                [- kb/2 + (m_a+h)*b +  w/2,      - ka/2 + (m_a+h)*a -  w/2     ],
                [  kb/2 -  m_a   *b +  w/2,        ka/2 -  m_a   *a -  w/2     ],
                [  kb/2 -  m_a   *b -  w/2,        ka/2 -  m_a   *a +  w/2     ],
            ]
        )

        match orientation:
            case 0:
                arrow = arrow_straight
            case 1:
                arrow = np.array([[-1,  0],
                                  [0,   1]]).dot(arrow_diagonal.T).T
            case 2:
                arrow = np.array([[-1,  0],
                                  [ 0, -1]]).dot(arrow_diagonal.T).T
            case 3:
                arrow = np.array([[ 1,  0],
                                  [ 0, -1]]).dot(arrow_straight.T).T
            case 4:
                arrow = np.array([[ 1,  0],
                                  [ 0, -1]]).dot(arrow_diagonal.T).T
            case 5:
                arrow = arrow_diagonal


        pg.draw.polygon(
            self.screen,
            color,
            [x, y] + arrow
        )
    
    def render(self, mode="men"):

        # Pythagorean triple in pixels
        a = 12
        b = 35
        c = math.floor(math.sqrt(a**2 + b**2) + 0.5)
        assert 0 < a
        assert a % 2 == 0
        assert a < b
        assert a**2 + b**2 == c**2

        # Scale
        k = 2
        assert 0 < k
        assert k % 2 == 0
        ka = k*a
        kb = k*b
        kc = k*c

        m_h = 2 # hexagon margin in pixels

        # Center
        y_offset = self.screen.get_height()//2 - self.space.radius*3*(ka + m_h)
        x_offset = self.screen.get_width()//2  - self.space.radius*(kb + 3*m_h)

        self.screen.fill((0, 0, 0)) # Fill the screen with black color

        # Render environment
        for y in range(self.space.array.shape[COORDINATE_Y]):
            for x in range(self.space.array.shape[COORDINATE_X]):
                if self.space.radius <= x + y <= 3*self.space.radius:
                    y_center = y_offset + y*2*(ka + m_h) + x*(ka + m_h)
                    x_center = x_offset                  + x*(kb + 3*m_h)
                    self.render_hexagon(y_center, x_center, ka, kb, m_h)
        
        # Render agents
        for agent_id in range(self.num_agents):
            if self.agent_types[agent_id] == self.ANGEL:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            y = self.space.agent_points[agent_id][COORDINATE_Y]
            x = self.space.agent_points[agent_id][COORDINATE_X]
            y_center = y_offset + y*2*(ka + m_h) + x*(ka + m_h)
            x_center = x_offset                  + x*(kb + 3*m_h)
            self.render_agent(y_center, x_center, ka, kb, a, b, self.space.agent_orientations[agent_id], color)

        pg.display.flip() # Update the display
