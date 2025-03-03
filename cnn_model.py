import torch
import torch.nn as nn
import torch.optim as optim

from database import train_loader, val_loader

class FaceCNN(nn.Module):
    def __init__(self, num_classes = 5749):
        super(FaceCNN, self).__init__()

        #First Convolutional Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1) #3 input channels for RGB, 64 filters
        self.bn1 = nn.BatchNorm2d(64) #normalization for stable training
        self.conv2 = nn.Conv2d(64,128, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2) #This reduces spatial size by half

        #Second Conv Block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(2, 2)      

        # Third Conv Block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling makes it flexible 

        # Fully Connected (Dense) Layer
        self.fc1 = nn.Linear(512 * 4 * 4, 1024) #Large fully connected layer
        self.dropout1 = nn.Dropout(0.5) #Regularization
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes) #Final Output Layer

    def forward(self, x):
        #Block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        #Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(torch.relu(self.bn2(self.conv4(x))))

        #Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  #Adaptive pooling

        #Flatten
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)     #No activation because CrossEntropyLoss applies softmax

        return x
    
#Create Model
num_classes = len(train_loader.dataset.dataset.classes)
model = FaceCNN(num_classes = num_classes)

#Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001) 

print(model)
