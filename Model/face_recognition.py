import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path_dir1 = './dataset/without_mask/'
path_dir2 = './dataset/with_mask/'

file_list1 = os.listdir(path_dir1)  # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)

file_list1_num = len(file_list1)
file_list2_num = len(file_list2)

file_num = file_list1_num + file_list2_num

# %% 이미지 전처리
num = 0;
all_img = np.float32(np.zeros((file_num, 224, 224, 3)))
all_label = np.float64(np.zeros((file_num, 1)))

for img_name in file_list1:
    img_path = path_dir1 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 0  # nomask
    num = num + 1

for img_name in file_list2:
    img_path = path_dir2 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 1  # mask
    num = num + 1