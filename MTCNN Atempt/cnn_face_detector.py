import torch
import torch.nn as nn
import torch.nn.functional as F

class PNet(nn.Module):
    """
    Proposal Network (P-Net):
    - Detects potential face regions in an image.
    - Performs classification (face/non-face).
    - Outputs bounding box proposals.
    """
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)  # Feature extraction
        self.prelu1 = nn.PReLU(10)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU(32)
        
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)  # Face classification
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)  # Bounding box regression

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        face_class = torch.sigmoid(self.conv4_1(x))  # Sigmoid for classification (face/non-face)
        bbox_reg = self.conv4_2(x)  # Bounding box regression
        return face_class, bbox_reg


class RNet(nn.Module):
    """
    Refinement Network (R-Net):
    - Takes proposals from P-Net.
    - Refines the face bounding box candidates.
    - Outputs refined classification and bounding box.
    """
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(28)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU(64)
        
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.prelu_fc = nn.PReLU(128)
        self.fc2_1 = nn.Linear(128, 1)  # Classification output
        self.fc2_2 = nn.Linear(128, 4)  # Bounding box regression

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.prelu_fc(self.fc1(x))
        face_class = torch.sigmoid(self.fc2_1(x))
        bbox_reg = self.fc2_2(x)
        return face_class, bbox_reg


class ONet(nn.Module):
    """
    Output Network (O-Net):
    - Final refinement step.
    - Generates highly accurate bounding boxes.
    - Outputs classification, bounding boxes.
    """
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU(128)
        
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.prelu_fc = nn.PReLU(256)
        self.fc2_1 = nn.Linear(256, 1)  # Classification output
        self.fc2_2 = nn.Linear(256, 4)  # Bounding box regression

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.prelu_fc(self.fc1(x))
        face_class = torch.sigmoid(self.fc2_1(x))
        bbox_reg = self.fc2_2(x)
        return face_class, bbox_reg


def face_detector():
    """Returns a complete face detection model with P-Net, R-Net, and O-Net."""
    return PNet(), RNet(), ONet()
