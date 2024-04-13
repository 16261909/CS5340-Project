import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

from config import *


def get_map(mask, PIL_mask):
    color_to_gray_map, gray_to_color_map = {}, {}
    PIL_mask = np.array(PIL_mask)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = tuple(mask[i, j])
            gray = PIL_mask[i, j]
            if color not in color_to_gray_map:
                gray_to_color_map[gray] = color
                color_to_gray_map[color] = gray

    return color_to_gray_map, gray_to_color_map


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
    if len(gray_masks.shape) == 2:
        color_masks = np.empty((gray_masks.shape[0], gray_masks.shape[1], 3), dtype=np.uint8)
        for j in range(gray_masks.shape[0]):
            for k in range(gray_masks.shape[1]):
                color_masks[j, k] = gray_to_color_map[gray_masks[j, k]]
    else:
        color_masks = np.empty((gray_masks.shape[0], gray_masks.shape[1], gray_masks.shape[2], 3), dtype=np.uint8)
        for i in range(gray_masks.shape[0]):
            for j in range(gray_masks.shape[1]):
                for k in range(gray_masks.shape[2]):
                    color_masks[i, j, k] = gray_to_color_map[gray_masks[i, j, k]]

    return color_masks


def convert_to_gray_mask(mask_frames, color_to_gray_map):
    flag = 0
    if (len(mask_frames.shape) != 4):
        flag = 1
        mask_frames = np.expand_dims(mask_frames, axis=0)


    gray_masks = np.empty((mask_frames.shape[0], mask_frames.shape[1], mask_frames.shape[2]), dtype=np.uint8)
    for i in range(mask_frames.shape[0]): # t
        for j in range(mask_frames.shape[1]): # w
            for k in range(mask_frames.shape[2]): # h
                color = tuple(mask_frames[i, j, k])
                gray_masks[i, j, k] = color_to_gray_map[color]

    if flag == 1:
        gray_masks = np.squeeze(gray_masks, axis=0)

    return gray_masks


def time_parameter(t):
    return max(np.power(time_base, t), time_min)


def s_parameter(t):
    return min(np.power(s_coefficient, t), time_min)


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

def generate_gif(image_path, ground_truth_path, mask_path, output_path):

    frames = []
    image_list = sorted(os.listdir(image_path))

    imgs = np.zeros((len(image_list), OutputResize[1], OutputResize[0], 3), dtype=np.uint8)
    gt = np.zeros((len(image_list), OutputResize[1], OutputResize[0], 3), dtype=np.uint8)
    mask = np.zeros((len(image_list), OutputResize[1], OutputResize[0], 3), dtype=np.uint8)

    for i in range(len(image_list)):
        mask[i] = cv2.resize(cv2.imread(os.path.join(mask_path, f"{i:05d}.png")), OutputResize, interpolation=cv2.INTER_NEAREST)
        imgs[i] = cv2.resize(cv2.imread(os.path.join(image_path, f"{i:05d}.jpg")), OutputResize, interpolation=cv2.INTER_NEAREST)
        gt[i] = cv2.resize(cv2.imread(os.path.join(ground_truth_path, f"{i:05d}.png")), OutputResize, interpolation=cv2.INTER_NEAREST)
        mask[i] = cv2.cvtColor(mask[i], cv2.COLOR_BGR2RGB)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        gt[i] = cv2.cvtColor(gt[i], cv2.COLOR_BGR2RGB)

        combined = Image.new('RGB', (imgs[i].shape[1] * 3, imgs[i].shape[0]))
        combined.paste(Image.fromarray(imgs[i]), (0, 0))
        combined.paste(Image.fromarray(gt[i]), (imgs[i].shape[1], 0))
        combined.paste(Image.fromarray(mask[i]), (imgs[i].shape[1] * 2, 0))

        frames.append(combined)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=100)


def generate_result(image_path, ground_truth_path, mask_path, output_path):

    frames = []
    image_list = sorted(os.listdir(image_path))

    imgs = np.zeros((len(image_list), OutputResize[1], OutputResize[0], 3), dtype=np.uint8)
    gt = np.zeros((len(image_list), OutputResize[1], OutputResize[0], 3), dtype=np.uint8)
    mask = np.zeros((len(image_list), OutputResize[1], OutputResize[0], 3), dtype=np.uint8)

    for i in range(len(image_list)):
        mask[i] = cv2.resize(cv2.imread(os.path.join(mask_path, f"{i:05d}.png")), OutputResize, interpolation=cv2.INTER_NEAREST)
        imgs[i] = cv2.resize(cv2.imread(os.path.join(image_path, f"{i:05d}.png")), OutputResize, interpolation=cv2.INTER_NEAREST)
        gt[i] = cv2.resize(cv2.imread(os.path.join(ground_truth_path, f"{i:05d}.png")), OutputResize, interpolation=cv2.INTER_NEAREST)
        mask[i] = cv2.cvtColor(mask[i], cv2.COLOR_BGR2RGB)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        gt[i] = cv2.cvtColor(gt[i], cv2.COLOR_BGR2RGB)

        combined = Image.new('RGB', (imgs[i].shape[1] * 3, imgs[i].shape[0]))
        combined.paste(Image.fromarray(imgs[i]), (0, 0))
        combined.paste(Image.fromarray(gt[i]), (imgs[i].shape[1], 0))
        combined.paste(Image.fromarray(mask[i]), (imgs[i].shape[1] * 2, 0))

        frames.append(combined)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=150)
