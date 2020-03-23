import pygame
from game import Game


class Cactus:
    def __init__(self):
        self.x = Game.width
        self.cactus = pygame.image.load('img/cactus.png').convert_alpha()
        self.sprite = pygame.sprite.Sprite()
        self.sprite.image = self.cactus
        self.rect = self.cactus.get_rect()
        self.rect.x = self.x
        self.rect.y = Game.height - 50 - self.rect.height
        self.hitbox = [self.rect.x,self.rect.y+5,41,45]

    def showCactus(self):
        Game.display.blit(self.cactus, self.rect)

