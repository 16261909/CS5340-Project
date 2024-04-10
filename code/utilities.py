import numpy as np
import cv2

time_base = 0.9
time_min = 0.3


def restore_color_mask(gray_masks, gray_to_color_map):
    color_masks = np.empty((gray_masks.shape[0], gray_masks.shape[1], 3), dtype=np.uint8)
    for i in range(gray_masks.shape[0]):
        for j in range(gray_masks.shape[1]):
            gray_value = gray_masks[i, j]
            color_masks[i, j] = gray_to_color_map[gray_value]

    return color_masks


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


def time_parameter(t):
    return max(np.power(time_base, t), time_min)


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
