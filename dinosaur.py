import pygame
from game import Game


class Dinosaur:
    image = pygame.image.load('img/dino.png').convert_alpha()

    def __init__(self):
        self.rect = self.image.get_rect()
        self.x = 10
        self.var = 50
        self.y = Game.height-self.var-self.rect.height
        self.hitbox = [self.x+5, self.y+20 ,45, 50]
        self.rect.x = self.x
        self.rect.y = self.y

    def jump(self):
        if self.var == 50:
            self.var += 150

    def show(self):
        Game.display.blit(self.image,self.rect)

    def updateVariables(self):
        self.x = 10
        self.y = Game.height - self.var - self.rect.height
        self.rect.x = self.x
        self.rect.y = self.y
        self.hitbox = [self.x + 5, self.y+20, 45, 50]