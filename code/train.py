import os
from time import sleep

import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image

from utilities import *
from resnet import *
from config import *

class CustomDataset(Dataset):
    def __init__(self, image_path, mask_path, image_transform=None, mask_transform=None, num_samples=1000):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = Image.open(self.image_path)
        mask = Image.open(self.mask_path)
        if self.image_transform and self.mask_transform:
            seed = np.random.randint(42)
            torch.manual_seed(seed)
            image = self.image_transform(image)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        return image, mask


mask_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10,  interpolation=Image.NEAREST),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor()
])

image_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10, interpolation=Image.BILINEAR),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor()
])


if __name__ == '__main__':

    np.random.seed(0)

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
        image_path = os.path.join(train_image_root, val_list[i] + '/00000.jpg')
        mask_path = os.path.join(train_mask_root, val_list[i] + '/00000.png')
        result_path = os.path.join(result_root, val_list[i])
        model_save_path = os.path.join(models_root, val_list[i] + '.pt')

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_NEAREST)
        # mask = cv2.resize(mask, (resize, resize), interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=0)
        mask, color_to_gray_map, gray_to_color_map = convert_to_gray_mask(mask)
        print('type_cnt:', len(color_to_gray_map))

        model = MyResNet(len(color_to_gray_map))
        mask = np.expand_dims(mask[0], axis=-1)

        input = np.concatenate((image, mask), axis=2)
        input = torch.tensor(input).permute(2, 0, 1).unsqueeze(0).float()

        dataset = CustomDataset(image_path, mask_path, image_transform=image_transforms, mask_transform=mask_transforms, num_samples=1000)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for image, mask in dataloader:
            print(image.shape, mask.shape)
            print_image(image[0].permute(1, 2, 0).numpy())
            print_image(mask[0].permute(1, 2, 0).numpy())

        for i in range(train_epoch):
            pass


        if not os.path.exists(models_root):
            os.makedirs(models_root)
        torch.save(model.state_dict(), model_save_path)

        sleep(1000000)




