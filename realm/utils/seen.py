import pygame as pg

from ..blessed_realm import BlessedRealm as Realm

SCREEN_LENGTH_Y = 480
SCREEN_LENGTH_X = 640


class Seen(Realm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize pygame
        pg.init()

        # Create a window
        self.screen = pg.display.set_mode((SCREEN_LENGTH_X, SCREEN_LENGTH_Y))
        
        pg.display.set_caption("Realm")
