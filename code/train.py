import os
from time import sleep

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch import nn
import torchvision.models as models

from utilities import *
from resnet import *
from config import *


if __name__ == '__main__':
    train_imageset_path = '../trainval/DAVIS/ImageSets/2017/train.txt'
    val_imageset_path = '../trainval/DAVIS/ImageSets/2017/val.txt'
    train_image_root = '../trainval/DAVIS/JPEGImages/480p/'
    train_mask_root = '../trainval/DAVIS/Annotations/480p/'
    testd_image_root = '../testd/DAVIS/JPEGImages/480p/'
    testd_mask_root = '../testd/DAVIS/Annotations/480p/'
    result_root = '../result/resnet/'
    models_root = '../models/'

    train_list = []
    val_list = []

    with open(train_imageset_path, 'r') as f:
        for line in f:
            train_list.append(line.strip())
    with open(val_imageset_path, 'r') as f:
        for line in f:
            val_list.append(line.strip())

    for i in range(len(train_list)):
        if i == 0:
            continue
        print(train_list[i])
        image_path = os.path.join(train_image_root, val_list[i])
        mask_path = os.path.join(train_mask_root, val_list[i] + '/00000.png')
        result_path = os.path.join(result_root, val_list[i])
        model_save_path = os.path.join(models_root, val_list[i] + '.pt')

        image_list = sorted(os.listdir(image_path))
        image = cv2.imread(os.path.join(image_path, image_list[0]))
        mask = cv2.imread(mask_path)

        image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (resize, resize), interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=0)
        mask, color_to_gray_map, gray_to_color_map = convert_to_gray_mask(mask)
        print('type_cnt:', len(color_to_gray_map))

        model = MyResNet(len(color_to_gray_map))
        mask = np.expand_dims(mask[0], axis=-1)
        input = np.concatenate((image, mask), axis=2)
        input = torch.tensor(input).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            output = model(input)

        sleep(1000000)



