import pygame
pygame.init()
clock = pygame.time.Clock()
width = 930
height = 700

display = pygame.display.set_mode((width,height))
gameExit = False

dino = pygame.image.load('img/dino.png').convert_alpha()

while not gameExit:
    display.fill((255,255,255))
    display.blit(dino,[10,550])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameExit = True

    pygame.display.update()
    clock.tick(30)
pygame.quit()
quit()