import os
import shutil
from tqdm import tqdm

import cv2
import numpy as np

from utilities import *
from mrf import *
from resnet import *

# configuration
ICM_iter = 4


if __name__ == '__main__':
    train_imageset_path = '../trainval/DAVIS/ImageSets/2017/train.txt'
    val_imageset_path = '../trainval/DAVIS/ImageSets/2017/val.txt'
    testd_imageset_path = '../testd/DAVIS/ImageSets/2017/test-dev.txt'

    train_flow_root = '../flow/trainval/'
    testd_flow_root = '../flow/test/'
    result_root = '../result/mrf/'
    train_image_root = '../trainval/DAVIS/JPEGImages/480p/'
    train_mask_root = '../trainval/DAVIS/Annotations/480p/'
    testd_image_root = '../testd/DAVIS/JPEGImages/480p/'
    testd_mask_root = '../testd/DAVIS/Annotations/480p/'
    rough_annotation_root = '../rough_annotation/osvos/'

    train_list = []
    val_list = []
    testd_list = []

    with open(train_imageset_path, 'r') as f:
        for line in f:
            train_list.append(line.strip())
    with open(val_imageset_path, 'r') as f:
        for line in f:
            val_list.append(line.strip())
    with open(testd_imageset_path, 'r') as f:
        for line in f:
            testd_list.append(line.strip())

    for i in range(len(val_list)):
        if i == 0:
            continue
        image_path = os.path.join(train_image_root, val_list[i])
        mask_path = os.path.join(train_mask_root, val_list[i] + '/00000.png')
        osvos_path = os.path.join(rough_annotation_root, val_list[i])
        result_path = os.path.join(result_root, val_list[i])
        flow_path = os.path.join(train_flow_root, val_list[i])

        image_list = sorted(os.listdir(image_path))
        image_list = image_list[:10]

        mask = cv2.imread(mask_path)
        mask = np.expand_dims(mask, axis=0)
        mask = np.tile(mask, (len(image_list), 1, 1, 1))
        imgs = np.zeros_like(mask)
        gray_imgs = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))

        osvos_mask = np.zeros_like(mask)

        mask, color_to_gray_map, gray_to_color_map = convert_to_gray_mask(mask)
        flo = np.zeros_like(mask)
        flo = np.expand_dims(flo, axis=-1)
        flo = np.tile(flo, (1, 1, 1, 2))

        type_cnt = len(color_to_gray_map)
        print('type_cnt:', type_cnt)
        # TODO: type_cnt > 2

        for i in range(len(image_list)):
            osvos_mask[i] = cv2.imread(os.path.join(osvos_path, f"{i:05d}.png"))

        osvos_mask, _, _ = convert_to_gray_mask(osvos_mask)
        osvos_mask[0] = mask[0]


        # for i in range(len(image_list)):
        #     imgs[i] = cv2.imread(os.path.join(image_path, f"{i:05d}.png"))
        #     gray_imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)

        for i in range(len(image_list) - 1):
            # flow_i_path = os.path.join(flow_path, f"{i + 1:06d}.flo")
            flo[i] = cv2.calcOpticalFlowFarneback(gray_imgs[i], gray_imgs[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flo[i] = filter_unreliable_flow(flo[i])

        init(flo)

        print('Start ICM...')

        for i in tqdm(range(ICM_iter)):
            for t in range(1, len(image_list)):
                for x in range(mask.shape[1]):
                    for y in range(mask.shape[2]):
                        idx = (t, x, y)
                        e_now = energy(mask, osvos_mask, idx)
                        diff_update(t, (x, y), -1)
                        mask[idx] = 1 - mask[idx]
                        diff_update(t, (x, y), 1)
                        e_nxt = energy(mask, osvos_mask, idx)
                        if e_now < e_nxt:
                            diff_update(t, (x, y), -1)
                            mask[idx] = 1 - mask[idx]
                            diff_update(t, (x, y), 1)

        # write
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path)

        for i in range(len(image_list)):
            result_i_path = result_path + f"/{i + 1:05d}.png"
            cv2.imwrite(result_i_path, restore_color_mask(mask[i], gray_to_color_map))

        break
