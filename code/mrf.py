import os
import shutil
from time import sleep

import cv2
import numpy as np

# configuration
theta_u = 1
theta_t = 1
theta_s = 1
ICM_iter = 10
osvos_posssibility = 0.99


# read images


def energy(mask: np.ndarray, osvos_mask: np.ndarray, idx):
    t = idx[0]
    x, y = idx[1], idx[2]

    e_u = -np.log(((1 - osvos_posssibility) if mask[idx] != osvos_mask[idx] else osvos_posssibility)) * theta_u
    e_t = 0
    e_s = 0
    return e_u + e_t + e_s


def inference():
    pass


def convert_to_gray_mask(mask_frames):
    color_to_gray_map = {}
    gray_to_color_map = {}
    current_label = 0

    gray_masks = np.empty((mask_frames.shape[0], mask_frames.shape[1], mask_frames.shape[2]), dtype=np.uint8)
    for i in range(mask_frames.shape[0]):
        for j in range(mask_frames.shape[1]):
            for k in range(mask_frames.shape[2]):
                color = tuple(mask_frames[i, j, k])
                if color not in color_to_gray_map:
                    color_to_gray_map[color] = current_label
                    gray_to_color_map[current_label] = color
                    current_label += 1
                gray_masks[i, j, k] = color_to_gray_map[color]

    return gray_masks, color_to_gray_map, gray_to_color_map

def restore_color_mask(gray_masks, gray_to_color_map):
    color_masks = np.empty((gray_masks.shape[0], gray_masks.shape[1], 3), dtype=np.uint8)
    for i in range(gray_masks.shape[0]):
        for j in range(gray_masks.shape[1]):
            gray_value = gray_masks[i, j]
            color_masks[i, j] = gray_to_color_map[gray_value]

    return color_masks


if __name__ == '__main__':
    train_imageset_path = '../trainval/DAVIS/ImageSets/2017/train.txt'
    val_imageset_path = '../trainval/DAVIS/ImageSets/2017/val.txt'
    testd_imageset_path = '../testd/DAVIS/ImageSets/2017/test-dev.txt'

    result_root = '../result/mrf/'
    train_image_root = '../trainval/DAVIS/JPEGImages/480p/'
    train_mask_root = '../trainval/DAVIS/Annotations/480p/'
    test_image_root = '../testd/DAVIS/JPEGImages/480p/'
    test_mask_root = '../testd/DAVIS/Annotations/480p/'
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

        images = sorted(os.listdir(image_path))

        images = images[:5]

        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path)

        mask = cv2.imread(mask_path)
        mask = np.expand_dims(mask, axis=0)
        mask = np.tile(mask, (len(images) - 1, 1, 1, 1))
        osvos_mask = np.zeros_like(mask)

        mask, color_to_gray_map, gray_to_color_map = convert_to_gray_mask(mask)

        type_cnt = len(color_to_gray_map)
        print(type_cnt)
        # TODO: type_cnt > 2

        for i in range(len(images) - 1):
            osvos_mask[i] = cv2.imread(os.path.join(osvos_path, f"{i + 1:05d}.png"))

        osvos_mask, _, _ = convert_to_gray_mask(osvos_mask)


        for i in range(ICM_iter):
            print(i)
            for j in range(len(images) - 1):
                for x in range(mask.shape[1]):
                    for y in range(mask.shape[2]):
                        idx = (j, x, y)
                        mask[idx] = 0
                        e0 = energy(mask, osvos_mask, idx)
                        mask[idx] = 1
                        e1 = energy(mask, osvos_mask, idx)
                        if e0 > e1:
                            mask[idx] = 0
                        else:
                            mask[idx] = 1


        for i in range(len(images) - 1):
            result_i_path = result_path + f"/{i + 1:05d}.png"
            cv2.imwrite(result_i_path, restore_color_mask(mask[i], gray_to_color_map))

        # count the number of colors in the mask
        print(mask.shape)
        sleep(100000)

        break