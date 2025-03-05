import torch
from torch.utils.data import DataLoader
from database import train_loader, val_loader
from cnn_model import FaceCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

#Create model and move to GPU
num_classes = len(train_loader.dataset.dataset.classes)
model = FaceCNN(num_classes = num_classes).to(device)

#Define Loss and Optimier
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        #Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backawrd Pass & Optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #Epoch Summary
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training Complete!")
torch.save(model.state_dict(), "face_cnn.pth")