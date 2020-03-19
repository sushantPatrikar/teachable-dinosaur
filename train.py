import cv2
import pandas as pd
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D
from keras.utils import to_categorical
from PIL import Image
global df
df = pd.DataFrame(columns=['Image','Action'])

def capture_do_nothing():
    global df
    action = 0
    camera = cv2.VideoCapture(0)
    exit = False
    while not exit:
        return_value, image = camera.read()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', image)
        count = 0
        if cv2.waitKey(1) & 0xFF == ord('s'):
            while count != 400:
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                gray1 = Image.fromarray(gray)
                gray1 = gray1.resize((50, 50))
                gray1 = gray1.transpose(Image.FLIP_LEFT_RIGHT)
                gray1 = np.array(gray1)
                df = df.append({'Image': gray1, 'Action': 0}, ignore_index=True)
                gray = Image.fromarray(gray)
                gray = gray.resize((50,50))
                gray = np.array(gray)
                df = df.append({'Image':gray,'Action':0},ignore_index=True)
                count+=1
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
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray1 = Image.fromarray(gray)
                gray1 = gray1.resize((50,50))
                gray1 = gray1.transpose(Image.FLIP_LEFT_RIGHT)
                gray1 = np.array(gray1)
                df = df.append({'Image': gray1, 'Action': 1}, ignore_index=True)
                gray = Image.fromarray(gray)
                gray = gray.resize((50, 50))
                gray = np.array(gray)
                df = df.append({'Image': gray, 'Action': 1}, ignore_index=True)
                count+=1
                return_value, image = camera.read()
            exit = True
    camera.release()
    cv2.destroyAllWindows()

def prepare_dataset():
    global df
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:,0]
    Y = df.iloc[:,1]
    x = []
    for i in range(1600):
        x.append(X[i])
    x = np.asarray(x)
    x = (x.astype(float)-128)/128
    x = np.reshape(x,(1600,50,50,1))
    y = to_categorical(Y)
    return x,y

def load_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),activation='relu',input_shape=(50,50,1)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def train(model,X,y):
    model.fit(X,y,batch_size=16,epochs=5)

def save_model(model):
    model.save_weights('weights.h5')

if __name__=='__main__':
    capture_do_nothing()
    time.sleep(1)
    capture_jump()
    time.sleep(1)
    X,y = prepare_dataset()
    model = load_model()
    train(model,X,y)
    save_model(model)