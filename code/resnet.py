import torch
import torchvision.models as models
import torch.nn as nn
from segmentation_models_pytorch import Unet


class MyResNet(nn.Module):
    def __init__(self, classes):
        super(MyResNet, self).__init__()
        self.resnet = Unet('resnet34', in_channels=4, classes = classes, encoder_weights='imagenet')


    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    model = MyResNet(2)
    print(model)

    model.eval()
    input = torch.randn(1, 4, 224, 224)
    output = model(input)

    print(output.shape, output[0, 0, 0, 0], output[0, 1, 0, 0])
