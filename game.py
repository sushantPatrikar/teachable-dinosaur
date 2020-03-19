import pygame
import random
import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D

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
    camera = cv2.VideoCapture(0)
    exit = False
    def __init__(self):
        self.model = self.load_model()
        self.model.load_weights('weights.h5')
        self.gameloop()
    def gameloop(self):
        while not self.gameExit:
            self.display.fill(self.white)
            pygame.draw.line(self.display, self.black, (0, self.height - 50), (self.width, self.height - 50))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gameExit = True
            if self.counter%(random.randrange(100,110))==0:
                from cactus import Cactus
                self.cactuses.append(Cactus())
            for c in self.cactuses:
                c.showCactus()
                c.x-=15
                if c.x <= -c.rect.width:
                    self.cactuses = self.cactuses[1:]
            pic = self.take_photo()
            result = self.predict(pic)
            print(result)
            if result==0:
                pass
            else:
                if self.y==50:
                    self.y+=150
            self.display.blit(self.dino, [10, self.height-self.y-self.dino_rect.height])
            pygame.display.update()
            if self.y>50:
                self.y-=10
            self.clock.tick(30)
            self.counter+=1
        pygame.quit()
        quit()

    def take_photo(self):
        return_value, image = self.camera.read()
        cv2.imshow('image', image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = Image.fromarray(gray)
        gray = gray.resize((50, 50))
        gray = np.array(gray)
        gray = np.resize(gray, (1, 50, 50, 1))
        (gray.astype(float) - 128) / 128
        return gray

    def load_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(50, 50, 1)))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self,pic):
        prob = self.model.predict(pic)
        result = np.argmax(prob)
        return result

if __name__=='__main__':
    g = Game()