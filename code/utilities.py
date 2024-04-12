import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import *


def filter_unreliable_flow(flo):
    reliable_flow = np.zeros_like(flo)
    # using sigma detection to filter out unreliable flow
    # FIXME: using other methods
    for i in range(flo.shape[0]):
        for d in range(2):
            mean = np.mean(flo[i, ..., d])
            std = np.std(flo[i, ..., d])
            reliable_flow[i, ..., d] = np.where(np.abs(mean - flo[i, ..., d]) < 2 * std, flo[i, ..., d], mean)
    return reliable_flow


def restore_color_mask(gray_masks, gray_to_color_map):
    color_masks = np.empty((gray_masks.shape[0], gray_masks.shape[1], 3), dtype=np.uint8)
    for i in range(gray_masks.shape[0]):
        for j in range(gray_masks.shape[1]):
            color_masks[i, j] = gray_to_color_map[gray_masks[i, j]]

    return color_masks


def convert_to_gray_mask(mask_frames):
    flag = 0
    if (len(mask_frames.shape) != 4):
        flag = 1
        mask_frames = np.expand_dims(mask_frames, axis=0)

    color_to_gray_map = {}
    gray_to_color_map = {}
    current_label = 0

    gray_masks = np.empty((mask_frames.shape[0], mask_frames.shape[1], mask_frames.shape[2]), dtype=np.uint8)
    for i in range(mask_frames.shape[0]): # t
        for j in range(mask_frames.shape[1]): # w
            for k in range(mask_frames.shape[2]): # h
                color = tuple(mask_frames[i, j, k])
                if color not in color_to_gray_map:
                    color_to_gray_map[color] = current_label
                    gray_to_color_map[current_label] = color
                    current_label += 1
                gray_masks[i, j, k] = color_to_gray_map[color]

    if flag == 1:
        gray_masks = np.squeeze(gray_masks, axis=0)

    return gray_masks, color_to_gray_map, gray_to_color_map


def time_parameter(t):
    return max(np.power(time_base, t), time_min)


def s_parameter(t):
    return np.power(s_coefficient, t)


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
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Invalid magic number {magic}")
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=width * height * 2)
        return data.reshape((height, width, 2))

def print_image(image, str='image'):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def print_images(images, str='image'):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def show_flo(flo):
    for i in range(flo.shape[0]):
        h, w = flo[i].shape[:2]
        y, x = np.mgrid[0:h:10, 0:w:10]
        fx, fy = flo[i][y, x].T
        lines = np.vstack([x.ravel(), y.ravel(), x.ravel() + fx.ravel(), y.ravel() + fy.ravel()]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        plt.figure(figsize=(10, 5))
        for line in lines:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='lime', linewidth=1)
        plt.axis('off')
        plt.title('Optical Flow Stripes')
        plt.show()


def resize_flo(flo):
    y_scale = Resize[0] / flo.shape[0]
    x_scale = Resize[1] / flo.shape[1]

    resized_flo = cv2.resize(flo, Resize, interpolation=cv2.INTER_NEAREST)

    resized_flo[..., 0] *= x_scale
    resized_flo[..., 1] *= y_scale

    return resized_flo