import cv2
import pandas as pd
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D,Dropout
from keras.utils import to_categorical
from PIL import Image
from skimage.transform import *

global df
df = pd.DataFrame(columns=['Image', 'Action'])


def capture_do_nothing():
    global df
    action = 0
    camera = cv2.VideoCapture(0)
    exit = False
    while not exit:
        return_value, image = camera.read()
        cv2.imshow('image', image)
        count = 0
        if cv2.waitKey(1) & 0xFF == ord('s'):
            while count != 400:
                im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                im = Image.fromarray(im)
                im = im.resize((50, 50))
                im = np.array(im)
                listt = augment_image(im)
                append_to_df(listt,0)
                count += 1
                return_value, image = camera.read()
            exit = True

    camera.release()
    cv2.destroyAllWindows()


def capture_jump():
    global df
    action = 1
    camera = cv2.VideoCapture(0)
    exit = False
    while not exit:
        return_value, image = camera.read()
        cv2.imshow('image', image)
        count = 0
        if cv2.waitKey(1) & 0xFF == ord('s'):
            while count != 400:
                im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                im = Image.fromarray(im)
                im = im.resize((50, 50))
                im = np.array(im)
                listt = augment_image(im)
                append_to_df(listt, 1)
                count += 1
                return_value, image = camera.read()
            exit = True
    camera.release()
    cv2.destroyAllWindows()

def flip_image(image):
    return np.fliplr(image)

def rotate_image(image):
    l_rotate = rotate(image,angle=40)
    r_rotate = rotate(image,angle=40)
    return l_rotate,r_rotate

def shift_image(image):
    transform = AffineTransform(translation=(-20,0))
    left_shift = warp(image,transform,mode="wrap")
    transform = AffineTransform(translation=(20,0))
    right_shift = warp(image,transform,mode="wrap")
    transform = AffineTransform(translation=(0,20))
    up_shift = warp(image,transform,mode="wrap")
    transform = AffineTransform(translation=(0,-20))
    down_shift = warp(image,transform,mode="wrap")
    return left_shift,right_shift,up_shift,down_shift

def blur_image(image):
    return cv2.GaussianBlur(image,(11,11),0)

def augment_image(image):
    flipped_image = flip_image(image)
    l_rotate,r_rotate = rotate_image(image)
    l_shift,r_shift,u_shift,d_shift = shift_image(image)
    blurred_image = blur_image(image)
    return [flipped_image,l_rotate,r_rotate,l_shift,r_shift,u_shift,d_shift,blurred_image]

def append_to_df(listt,action):
    global df
    for image in listt:
        df = df.append({'Image':image,'Action':int(action)},ignore_index=True)

def prepare_dataset():
    global df
    # df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, 0]
    Y = df.iloc[:, 1]
    x = []
    for i in range(6400):
        print(i)
        x.append(X[i])
    x = np.asarray(x)
    x = (x.astype(float) - 128) / 128
    x = np.reshape(x, (6400, 50, 50, 1))
    y = to_categorical(Y)
    return x, y



def load_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(50, 50, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model, X, y):
    model.fit(X, y, batch_size=32, epochs=3,shuffle=True)


def save_model(model):
    model.save_weights('weights.h5')


if __name__ == '__main__':
    capture_do_nothing()
    time.sleep(1)
    capture_jump()
    time.sleep(1)
    X, y = prepare_dataset()
    model = load_model()
    train(model, X, y)
    save_model(model)
