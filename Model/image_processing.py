# from PIL import Image
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

imagePath = "../Dataset"
image_w = 28
image_h = 28
categories = ["with_mask", "without_mask"]
nb_classes = len(categories)
#pixels = image_w * image_h * 3
X = []
Y = []
for idx, cate in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = imagePath+'/'+cate+'/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
xy = (X_train, X_test, Y_train, Y_test)

np.savetxt('data.txt', xy, fmt = '%s', delimiter = ',')
np.save("../Dataset", xy)