import os
import torch

from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.models import VGG16_Weights

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FACE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "standard_face.jpg"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "standard_face_feature.pt"
)

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

model = models.vgg16(
    weights=VGG16_Weights.DEFAULT
)

model.eval()

img = Image.open(FACE_PATH).convert("RGB")

x = transform(img).unsqueeze(0)

with torch.no_grad():

    feature = model(x)

torch.save(feature, FEATURE_PATH)

print("标准人脸特征已保存：")
print(FEATURE_PATH)
print("特征维度：", feature.shape)