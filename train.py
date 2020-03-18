import cv2
import time

def capture_do_nothing():
    camera = cv2.VideoCapture(0)
    exit = False
    while not exit:
        return_value, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', gray)
        count = 0
        if cv2.waitKey(1) & 0xFF == ord('s'):
            while count != 50:
                cv2.imwrite('do_nothing\\photo_{}.png'.format(count),image)
                print('captured photo',count)
                count+=1
                return_value, image = camera.read()
            exit = True

    camera.release()
    cv2.destroyAllWindows()

def capture_jump():
    camera = cv2.VideoCapture(0)
    exit = False
    while not exit:
        return_value, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', gray)
        count = 0
        if cv2.waitKey(1) & 0xFF == ord('s'):
            while count != 50:
                cv2.imwrite('jump\\photo_{}.png'.format(count),image)
                print('captured photo',count)
                count+=1
                return_value, image = camera.read()
            exit = True
    camera.release()
    cv2.destroyAllWindows()
