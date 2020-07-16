from scipy.io import loadmat, savemat
import numpy as np
from PIL import Image
import os

def resize_images():
    with open('lists/train.txt') as f_class:
        l = []
        t = []
        r = []
        b = []
        for line in f_class:
            a = loadmat('annotations/annotations-mat/' + line.rstrip('.jpg\n'))
            l = a['bbox'][0,0][0][0,0]
            t = a['bbox'][0,0][1][0,0]
            r = a['bbox'][0,0][2][0,0]
            b = a['bbox'][0,0][3][0,0]

            img = Image.open('images/' + line.rstrip('\n'))
            img_cropped = img.crop((l, t, r, b))
            img_resized = img_cropped.resize((224, 224))
            os.makedirs('images_resized/' + line.split('/')[0], exist_ok=True)
            img_resized.save('images_resized/' + line.rstrip('\n'))

    with open('lists/test.txt') as f_class:
        l = []
        t = []
        r = []
        b = []
        for line in f_class:
            a = loadmat('annotations/annotations-mat/' + line.rstrip('.jpg\n'))
            l = a['bbox'][0,0][0][0,0]
            t = a['bbox'][0,0][1][0,0]
            r = a['bbox'][0,0][2][0,0]
            b = a['bbox'][0,0][3][0,0]

            img = Image.open('images/' + line.rstrip('\n'))
            img_cropped = img.crop((l, t, r, b))
            img_resized = img_cropped.resize((224, 224))
            os.makedirs('images_resized/' + line.split('/')[0], exist_ok=True)
            img_resized.save('images_resized/' + line.rstrip('\n'))

def create_arrays():
    X_train = []
    y_train = []
    with open('lists/train.txt') as f_class:
        for line in f_class:
            img = Image.open('images_resized/' + line.rstrip('\n'))
            img_arr = np.array(img).transpose(2, 0, 1)
            X_train.append(img_arr)
            y_train.append(int(line[:3]) - 1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    savemat('train.mat', {'X': X_train, 'y': y_train})

    X_test = []
    y_test = []
    with open('lists/test.txt') as f_class:
        for line in f_class:
            img = Image.open('images_resized/' + line.rstrip('\n'))
            img_arr = np.array(img).transpose(2, 0, 1)
            X_test.append(img_arr)
            y_test.append(int(line[:3]) - 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    savemat('test.mat', {'X': X_test, 'y': y_test})

if __name__ == '__main__':
    resize_images()
    create_arrays()

