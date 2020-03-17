import pygame
import random
class Game:
    pygame.init()
    clock = pygame.time.Clock()
    width = 1024
    height = 700
    black = (0,0,0)
    white = (255,255,255)
    display = pygame.display.set_mode((width,height))
    gameExit = False
    dino = pygame.image.load('img/dino.png').convert_alpha()
    cactus = pygame.image.load('img/cactus.png').convert_alpha()
    dino_rect = dino.get_rect()
    cactus_rect = cactus.get_rect()
    cactuses = []
    counter = 0
    y = 50
    def __init__(self):
        self.gameloop()
    def gameloop(self):
        while not self.gameExit:
            self.display.fill(self.white)
            pygame.draw.line(self.display, self.black, (0, self.height - 50), (self.width, self.height - 50))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gameExit = True
                if event.type==pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.y==50:
                        self.y +=150
            if self.counter%(random.randrange(100,110))==0:
                from cactus import Cactus
                self.cactuses.append(Cactus())
            for c in self.cactuses:
                c.showCactus()
                c.x-=15
                if c.x <= -c.rect.width:
                    self.cactuses = self.cactuses[1:]
            self.display.blit(self.dino, [10, self.height-self.y-self.dino_rect.height])
            pygame.display.update()
            if self.y>50:
                self.y-=10
            self.clock.tick(30)
            self.counter+=1
        pygame.quit()
        quit()

if __name__=='__main__':
    g = Game()