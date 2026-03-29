import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image

#  Dataset dizini
dataset_path = r"C:/Users/sedat/OneDrive/Masaüstü/deep learning projeler/archive/Brain_Data_Organised"

#  Veri Dönüştürme ve Artırma
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Grayscale dönüşümü
    transforms.RandomHorizontalFlip(),         # Yansıtma
    transforms.RandomRotation(10),             # Dönme
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # Grayscale için tek kanal normalize
                         std=[0.5])
])

#  Klasördeki tüm fotoğrafları al
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 'Normal' ve 'Stroke' klasörlerine bak
        for label, folder in enumerate(['Normal', 'Stroke']):
            folder_path = os.path.join(root_dir, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):  # Dosya uzantıları
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # 'L' mode for grayscale (tek kanal)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

#  Dataset'i yükle
dataset = CustomImageDataset(root_dir=dataset_path, transform=transform)

#  Train/Validation ayır (test veri seti burada validation olarak kullanılacak)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

#  Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  CNN Modeli
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 kanal için
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ➕ Dinamik FC boyutu hesaplamak için dummy forward (224 → 28 oluyor)
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            nn.ReLU(),
            self.pool
        )
        self._get_conv_output()
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def _get_conv_output(self):
        x = torch.randn(1, 1, 224, 224)  # 1 kanal, 224x224 boyutunda dummy giriş
        x = self.convs(x)
        self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = BrainTumorCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

#  EarlyStopping
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

#  Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs=7, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []  # Doğruluk değerleri için listeler

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{max_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        early_stopping(avg_val_loss)
        scheduler.step(avg_val_loss)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies

#  Eğitim Başlat
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    max_epochs=7,
    patience=3
)

#  Sonuçları Görselleştir
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Loss grafiği
ax[0].plot(train_losses, label='Train Loss')
ax[0].plot(val_losses, label='Validation Loss')
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Model Loss Curve")
ax[0].legend()

# Doğruluk grafiği
ax[1].plot(train_accuracies, label='Train Accuracy')
ax[1].plot(val_accuracies, label='Validation Accuracy')
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy (%)")
ax[1].set_title("Model Accuracy Curve")
ax[1].legend()

# Modelin ağırlıklarını kaydet (GPU'da eğitilmiş olsa bile otomatik işler)
torch.save(model.state_dict(), 'brain_tumor_model_weights.pth')

# Ek bilgi: Eğer optimizer'ın durumunu da kaydetmek isterseniz:
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 7,
    'val_accuracy': 97.06
}
torch.save(checkpoint, 'checkpoint.pth')  # Opsiyonel

plt.show()
