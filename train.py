import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# --- 1. НАСТРОЙКИ ---
# ВАЖНО: Замените обратные слеши на прямые или используйте r"", чтобы избежать ошибок в Python
DATA_DIR = r"E:\3 курс\Lor_Ai\Otoscopic_Data"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # У нас 5 папок

# Проверяем, доступна ли видеокарта
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# --- 2. ПРЕДОБРАБОТКА ДАННЫХ (АУГМЕНТАЦИЯ) ---
# Медицинские снимки нужно нормализовать и немного изменять при обучении,
# чтобы модель не запоминала картинки наизусть (overfitting)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Стандартный размер для ResNet
        transforms.RandomHorizontalFlip(),  # Отражение по горизонтали
        transforms.RandomRotation(15),  # Поворот на 15 градусов
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Изменение яркости/контраста
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Стандартные значения ImageNet
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. ЗАГРУЗКА ДАННЫХ ---
# Загружаем весь датасет
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms['train'])
class_names = full_dataset.classes
print(f"Найдены классы: {class_names}")

# Разделяем на Train (80%) и Validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Для валидации отключаем аугментацию
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. СОЗДАНИЕ МОДЕЛИ (TRANSFER LEARNING) ---
# Используем ResNet18 - она быстрая и точная
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Замораживаем все слои, кроме последнего
for param in model.parameters():
    param.requires_grad = False

# Заменяем последний слой (fc) под наши 5 классов
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# --- 5. ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАТОР ---
criterion = nn.CrossEntropyLoss()
# Обучаем только параметры последнего слоя
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# --- 6. ЦИКЛ ОБУЧЕНИЯ ---
def train_model():
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # Фаза обучения
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = corrects.double() / len(train_dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Фаза валидации
        model.eval()
        val_running_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_lor_model.pth')
            print("Модель сохранена!")


if __name__ == '__main__':
    train_model()