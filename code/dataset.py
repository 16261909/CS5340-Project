
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image

from utilities import *

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
        if mask.mode == "L":
            print("Mask is single channel")
        else:
            print("Mask is not single channel")

        if self.image_transform and self.mask_transform:
            seed = np.random.randint(42)
            torch.manual_seed(seed)
            image_transformed = self.image_transform(image)
            torch.manual_seed(seed)
            mask_transformed = self.mask_transform(mask)

            print(image_transformed.shape, mask_transformed.shape)

            return image_transformed, mask_transformed
        else:
            return image, mask

train_mask_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10,  interpolation=Image.NEAREST),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomCrop(Resize),
    transforms.ToTensor()
])

train_image_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10, interpolation=Image.BILINEAR),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomCrop(Resize),
    transforms.ToTensor()
])

val_mask_transforms = transforms.Compose([
    transforms.Resize(Resize, interpolation=Image.NEAREST),
    transforms.ToTensor()
])

val_image_transforms = transforms.Compose([
    transforms.Resize(Resize, interpolation=Image.NEAREST),
    transforms.ToTensor()
])