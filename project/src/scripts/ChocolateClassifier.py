import torch
import torch.nn as nn
import torch.nn.functional as F

class ChocolateClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(ChocolateClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # → 112×112
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # → 56×56
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # → 28×28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # → 14×14
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # 224 → 112
        x = self.conv2(x)  # → 56
        x = self.conv3(x)  # → 28
        x = self.conv4(x)  # → 14
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

