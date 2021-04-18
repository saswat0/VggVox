import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class VggVox(nn.Module):

    def __init__(self, num_classes=2):
        super(VggVox, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.bn6 = nn.BatchNorm2d(num_features=4096)
        self.bn7 = nn.BatchNorm1d(num_features=1024)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mpool5 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
        
        self.fc6 = nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9, 1))
        self.fc7 = nn.Linear(in_features=4096, out_features=1024)
        self.fc8 = nn.Linear(in_features=1024, out_features=num_classes)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.mpool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.mpool5(x)
        x = self.relu(self.bn6(self.fc6(x)))
        
        _, _, _, W = x.size()
        self.apool6 = nn.AvgPool2d(kernel_size=(1, W))
        x = self.apool6(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn7(self.fc7(x)))
        x = self.fc8(x)
        
        if self.training:
            return x
        
        else:
            return self.softmax(x)