import os
import shutil
from tqdm import tqdm

import cv2
import numpy as np

# configuration
theta_u = 1
theta_t = 1
theta_s = 1
ICM_iter = 4
e_u_true_possibility = 0.99
log_e_u_true_possibility = np.log(e_u_true_possibility)
e_u_false_possibility = 1 - e_u_true_possibility
log_e_u_false_possibility = np.log(e_u_false_possibility)


flow_range = 3
flow_threshold = 100
time_base = 0.9
time_min = 0.3

flow_out_of_uncalculated_err = -20000
flow_out_of_range_err = -10000


def time_parameter(t):
    return max(np.power(time_base, t), time_min)


def filter_unreliable_flow(flow):
    flow_magnitude = np.linalg.norm(flow, axis=2)
    reliable_flow = np.zeros_like(flow)
    reliable_flow[flow_magnitude < flow_threshold] = flow[flow_magnitude < flow_threshold]
    return reliable_flow


def show_flow(flow):
    flow_magnitude = np.linalg.norm(flow, axis=2)
    flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = flow_angle * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', bgr)
    cv2.waitKey(0)


def get_positions(flo, precomputed_positions, pos, t):

    if (precomputed_positions[t, pos[0], pos[1]] != flow_out_of_uncalculated_err).all():
        return precomputed_positions[t, pos[0], pos[1], 0], precomputed_positions[t, pos[0], pos[1], 1]

    # [-flow_range, -flow_range + 1, ..., -1, 1, ..., flow_range - 1, flow_range]
    precomputed_positions[t, pos[0], pos[1], 0] = np.full((flow_range, 2), flow_out_of_range_err)
    precomputed_positions[t, pos[0], pos[1], 1] = np.full((flow_range, 2), flow_out_of_range_err)
    cur_pos = list(pos)
    # Loop from 0 to flow_range
    for offset in range(min(flow_range, precomputed_positions.shape[0] - t - 1)):
        flow = flo[t + offset]
        dx, dy = flow[cur_pos[0], cur_pos[1]]
        cur_pos = [round(cur_pos[0] + dx), round(cur_pos[1] + dy)]
        if cur_pos[0] < 0 or cur_pos[0] >= flo.shape[1] or cur_pos[1] < 0 or cur_pos[1] >= flo.shape[2]:
            break
        precomputed_positions[t, pos[0], pos[1], 1, offset] = [cur_pos[0], cur_pos[1]]
        precomputed_positions[t + offset + 1, cur_pos[0], cur_pos[1], 0, offset] = [pos[0], pos[1]]

    cur_pos = list(pos)
    # Loop from 0 to -flow_range
    for offset in range(min(flow_range, t - 1)):
        flow = flo[t - offset]
        dx, dy = flow[cur_pos[0], cur_pos[1]]
        cur_pos = [round(cur_pos[0] - dx), round(cur_pos[1] - dy)]
        if cur_pos[0] < 0 or cur_pos[0] >= flo.shape[1] or cur_pos[1] < 0 or cur_pos[1] >= flo.shape[2]:
            break
        precomputed_positions[t, pos[0], pos[1], 0, offset] = [cur_pos[0], cur_pos[1]]
        precomputed_positions[t - offset - 1, cur_pos[0], cur_pos[1], 1, offset] = [pos[0], pos[1]]

    return precomputed_positions[t, pos[0], pos[1], 0], precomputed_positions[t, pos[0], pos[1], 1]


def energy(mask: np.ndarray, osvos_mask: np.ndarray, idx):

    global precomputed_positions, diff_between_masks

    if precomputed_positions is None or diff_between_masks is None:
        raise ValueError("Global variables precomputed_positions and diff_between_masks must be initialized before calling this function.")

    t = idx[0]
    x, y = idx[1], idx[2]
    e_u, e_t, e_s = 0, 0, 0

    e_u += -log_e_u_false_possibility if mask[idx] != osvos_mask[idx] else -log_e_u_true_possibility

    # -1 -> -flow_range
    for dt in range(min(flow_range, t - 1)):
        e_t += time_parameter(t - dt - 1) * time_parameter(t) * diff_between_masks[t - dt - 1, t] * diff_between_masks[t - dt - 1, t]

    # 1 -> flow_range
    for dt in range(min(flow_range, mask.shape[0] - t - 1)):
        e_t += time_parameter(t + dt + 1) * time_parameter(t) * diff_between_masks[t, t + dt + 1] * diff_between_masks[t, t + dt + 1]

    return theta_u * e_u + theta_t * e_t + theta_s * e_s


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


def diff_update(t, pos, dv):
    x, y = pos[0], pos[1]

    global mask

    for dt in range(min(flow_range, t - 1)):
        if precomputed_positions[t, x, y, 0, dt, 0] == flow_out_of_range_err:
            break
        if mask[t - dt - 1, precomputed_positions[t, x, y, 0, dt, 0], precomputed_positions[t, x, y, 0, dt, 1]] != mask[t, x, y]:
            diff_between_masks[t - dt - 1, t] += dv

    for dt in range(min(flow_range, mask.shape[0] - t - 1)):
        if precomputed_positions[t, x, y, 1, dt, 0] == flow_out_of_range_err:
            break
        if mask[t + dt + 1, precomputed_positions[t, x, y, 1, dt, 0], precomputed_positions[t, x, y, 1, dt, 1]] != mask[t, x, y]:
            diff_between_masks[t, t + dt + 1] += dv


def init(flo):

    global precomputed_positions, diff_between_masks

    print('Start init...')

    precomputed_positions = np.empty((mask.shape[0], mask.shape[1], mask.shape[2], 2, flow_range, 2), dtype=tuple)
    precomputed_positions.fill(flow_out_of_uncalculated_err)

    for t in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            for y in range(mask.shape[2]):
                neg_ret, pos_ret = get_positions(flo, precomputed_positions, (x, y), t)
                precomputed_positions[t, x, y, 0] = neg_ret
                precomputed_positions[t, x, y, 1] = pos_ret

    diff_between_masks = np.zeros((mask.shape[0], mask.shape[0]))

    for t in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            for y in range(mask.shape[2]):
                diff_update(t, (x, y), 1)


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
