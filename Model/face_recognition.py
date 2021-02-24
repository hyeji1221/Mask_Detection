import os
import numpy as np
import cv2

imagePath = "../Dataset"
image_w = 28
image_h = 28
categories = ["with_mask", "without_mask"]
filelistnum1=len(categories[1])
filelistnum2=len(categories[0])
# %% 이미지 전처리
label=[0 for i in range(filelistnum1+filelistnum2)]
num=0
all_label=[]
for img_name in categories[1]:
    img_path =imagePath +'/'+ img_name+'/'
    for top,dir,f in os.walk(img_path):
        for filename in f:
            img=cv2.imread(img_path+filename)
            img=cv2.resize(img,None,fx=image_w/img.shape[1],fy=image_h/img.shape[0])
            all_label[num] = 0  # nomask
            num=num+1
num=0
for img_name in categories[0]:
    img_path = imagePath+'/' + img_name+'/'
    for top, dir, f in os.walk(img_path):
        for filename in f:
            img = cv2.imread(img_path + filename)
            img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            all_label[num] = 1  # mask
            num = num + 1

np.save("Dataset.npy", all_label)