import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from pillow_heif import register_heif_opener
from line_profiler import profile
from helper_funcs import clear_lines

# Register HEIF opener
register_heif_opener()

profile.disable()

class CustomDataset(Dataset):
    @profile
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    @profile
    def __len__(self):
        return len(self.data)

    @profile
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        points = self.data.iloc[idx, 1:].values.astype('float32')

        image = Image.open(img_path)#.convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(points)

class CNN(nn.Module):
    @profile
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 8)
        self.relu = nn.ReLU()

    @profile
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x)))
        x = self.pool(self.relu(self.conv7(x)))
        x = x.view(-1, 64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@profile
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, points in tqdm(train_loader, desc="Training"):
        images, points = images.to(device), points.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, points)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

@profile
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, points in tqdm(val_loader, desc="Validating"):
            images, points = images.to(device), points.to(device)
            outputs = model(images)
            loss = criterion(outputs, points)
            running_loss += loss.item()
    return running_loss / len(val_loader)

@profile
def main():
    # Settings
    batch_size = 128

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Grayscale(num_output_channels=1),  
    ])
    transform2 = transforms.Compose([
        transforms.ColorJitter(),
    ])

    # Initialize the model, loss function, and optimizer
    model = CNN().to(device)
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create datasets
    dataset = CustomDataset('corners.csv', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"{train_size:,} items in train dataset.")
    print(f"{val_size:,} items in val dataset.")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=(train_size if batch_size == 'all' else batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=(val_size if batch_size == 'all' else batch_size), shuffle=False)

    train_set = [item for item in tqdm(train_loader, desc="Loading train dataset")]
    val_set = [item for item in tqdm(val_loader, desc="Loading val dataset")]    

    # Training loop
    num_epochs = 1_000
    for epoch in range(num_epochs):
        train_set = [transform2(item) for item in train_set]
        train_loss = train(model, train_set, criterion, optimizer, device)
        val_loss = validate(model, val_set, criterion, device)
        clear_lines(2)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model
        torch.save(model, 'cornerModels/model.pt')

if __name__ == "__main__":
    main()