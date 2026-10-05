import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Настройки ---
DATA_DIR = r"E:\3 курс\Lor_Ai\Otoscopic_Data"
MODEL_PATH = 'best_lor_model.pth'
BATCH_SIZE = 16
device = torch.device("cpu")

# --- Трансформации (как при валидации) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Загрузка данных ---
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = full_dataset.classes
loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Загрузка модели ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Прогон по всему датасету ---
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Предсказано')
plt.ylabel('Истинно')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# --- Отчет по метрикам ---
print("\n=== Отчет по классификации ===\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))