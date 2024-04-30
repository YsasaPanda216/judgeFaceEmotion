import torch.nn as nn
from torchvision.models import resnet18

class Resnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 3)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h
