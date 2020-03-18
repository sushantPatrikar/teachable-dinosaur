import cv2
import pandas as pd
import time
import numpy as np

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
            while count != 50:
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                gray = np.resize(gray,(50,50))
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
            while count != 50:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.resize(gray, (50, 50))
                df = df.append({'Image': gray, 'Action': 1}, ignore_index=True)
                count+=1
                return_value, image = camera.read()
            exit = True
    camera.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    capture_do_nothing()
    time.sleep(1)
    capture_jump()
    time.sleep(1)