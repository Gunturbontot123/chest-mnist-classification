# model.py
import torch
import torch.nn as nn
from torchvision.models import densenet121

class DenseNet121Binary(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        self.model = densenet121(pretrained=pretrained)

        # ubah input pertama ke grayscale
        self.model.features.conv0 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # ubah classifier jadi 1 neuron
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

    def forward(self, x):
        return self.model(x)
