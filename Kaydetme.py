import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms

#  Model Tanımı
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convs = nn.Sequential(
            self.conv1, nn.ReLU(), self.pool,
            self.conv2, nn.ReLU(), self.pool,
            self.conv3, nn.ReLU(), self.pool
        )

        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 224, 224)
            x = self.convs(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#  Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Modeli yükle
model = BrainTumorCNN().to(device)
try:
    model.load_state_dict(torch.load('brain_tumor_model_weights.pth', map_location=device))
    model.eval()
    print("✅ Model başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {e}")
    exit()

#  Görüntü dönüştürme (ön işleme)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#  Test klasörü
test_folder = r"C:\Users\sedat\OneDrive\Masaüstü\deep learning projeler\archive\Brain_Data_Organised\Test_Image"

#  Tahminleri yap
if not os.path.exists(test_folder):
    print("❌ Belirtilen test klasörü bulunamadı.")
    exit()

print("🔎 Görseller işleniyor...")

for filename in os.listdir(test_folder):
    file_path = os.path.join(test_folder, filename)

    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            image = Image.open(file_path).convert("L")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)

            sınıf = "Normal" if predicted.item() == 0 else "Stroke"
            print(f'{filename} - Tahmin Edilen Sınıf: {sınıf}')
        except Exception as img_err:
            print(f"{filename} - Görüntü işlenirken hata: {img_err}")