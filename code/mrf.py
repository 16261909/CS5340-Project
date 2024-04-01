import os
import shutil
from time import sleep
from PIL import Image

import cv2
import numpy as np

# configuration
theta_u = 1
theta_t = 1
theta_s = 1
ICM_iter = 10
osvos_posssibility = 0.99
flow_range = 3
flow_out_of_range_err = (-10000, -10000)

def get_positions(flo, pos, t):
    ret = [flow_out_of_range_err] * (2 * flow_range + 1)
    cur_pos = list(pos)

    # Loop from 0 to flow_range
    for offset in range(1, min(flow_range + 1, flo.shape[0] - t)):
        flow = flo[t + offset]
        dx, dy = flow[cur_pos[1], cur_pos[0]]
        cur_pos = (cur_pos[0] + dx, cur_pos[1] + dy)
        ret[flow_range + offset] = (round(cur_pos[0]), round(cur_pos[1]))

    # Reset position
    cur_pos = list(pos)

    # Loop from 0 to -flow_range
    for offset in range(-1, -min(flow_range + 1, t + 1), -1):
        flow = flo[t + offset]
        dx, dy = flow[cur_pos[1], cur_pos[0]]
        cur_pos = (cur_pos[0] - dx, cur_pos[1] - dy)
        ret[flow_range + offset] = (round(cur_pos[0]), round(cur_pos[1]))

    print(ret)
    return ret


def energy(mask: np.ndarray, osvos_mask: np.ndarray, flo: np.ndarray, idx):
    t = idx[0]
    x, y = idx[1], idx[2]

    positions = get_positions(flo, (x, y), t)
    print(positions, t)

    e_u = -np.log(((1 - osvos_posssibility) if mask[idx] != osvos_mask[idx] else osvos_posssibility)) * theta_u
    e_t = 0
    e_s = 0

    sleep(1000000)
    return e_u + e_t + e_s


def inference():
    pass

def read_flo_file(file_path):
    print(file_path)
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Invalid magic number {magic}")
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=width * height * 2)
        return data.reshape((height, width, 2))

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
        image_list = image_list[:3]

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
        print(type_cnt)
        # TODO: type_cnt > 2

        for i in range(len(image_list)):
            osvos_mask[i] = cv2.imread(os.path.join(osvos_path, f"{i + 1:05d}.png"))

        osvos_mask, _, _ = convert_to_gray_mask(osvos_mask)

        for i in range(len(image_list)):
            imgs[i] = cv2.imread(os.path.join(image_path, image_list[i]))
            gray_imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)


        for i in range(len(image_list) - 1):
            # flow_i_path = os.path.join(flow_path, f"{i + 1:06d}.flo")
            print(gray_imgs[i].shape, gray_imgs[i+1].shape)
            flo[i] = cv2.calcOpticalFlowFarneback(gray_imgs[i], gray_imgs[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

        print(flo.shape)

        for i in range(ICM_iter):
            print(i)
            for j in range(len(image_list)):
                for x in range(mask.shape[1]):
                    for y in range(mask.shape[2]):
                        idx = (j, x, y)
                        mask[idx] = 0
                        e0 = energy(mask, osvos_mask, flo, idx)
                        mask[idx] = 1
                        e1 = energy(mask, osvos_mask, flo, idx)
                        if e0 < e1:
                            mask[idx] = 0
                        else:
                            mask[idx] = 1

        # write
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path)

        for i in range(len(image_list)):
            result_i_path = result_path + f"/{i + 1:05d}.png"
            cv2.imwrite(result_i_path, restore_color_mask(mask[i], gray_to_color_map))

        # count the number of colors in the mask
        print(mask.shape)
        sleep(100000)

        break