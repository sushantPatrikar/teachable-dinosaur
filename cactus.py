import pygame
from game import Game


class Cactus:
    def __init__(self):
        self.x = Game.width
        self.cactus = pygame.image.load('img/cactus.png').convert_alpha()
        self.rect = self.cactus.get_rect()

    def showCactus(self):
        Game.display.blit(self.cactus, [self.x, Game.height - 50 - self.rect.height])
