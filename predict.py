import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# --- Настройки ---
MODEL_PATH = 'best_lor_model.pth'
CLASSES = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']
device = torch.device("cpu")

# 🔧 Порог уверенности: если ниже — "не знаю"
CONFIDENCE_THRESHOLD = 0.90    # 90%
MARGIN_THRESHOLD = 0.30        # разрыв между 1-м и 2-м местом (30%)

# --- Загрузка модели ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Трансформации ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"\n❌ Файл не найден: {image_path}\n")
        return
    except Exception as e:
        print(f"\n❌ Ошибка при открытии: {e}\n")
        return

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Сортируем вероятности по убыванию
    sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)
    top1_prob = sorted_probs[0].item()
    top2_prob = sorted_probs[1].item()
    margin = top1_prob - top2_prob
    predicted_class = CLASSES[sorted_idx[0].item()]

    print(f"\n{'='*60}")
    print(f"📁 Файл: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # 🎯 Логика определения "не знаю"
    if top1_prob < CONFIDENCE_THRESHOLD:
        print(f"❓ Я НЕ ЗНАЮ, ЧТО ЭТО")
        print(f"   Уверенность слишком низкая: {top1_prob*100:.2f}% (порог: {CONFIDENCE_THRESHOLD*100:.0f}%)")
        print(f"   Возможно, это заболевание, которого нет в моём датасете.")
        print(f"   Или фото плохого качества / не является отоскопическим снимком.")
    elif margin < MARGIN_THRESHOLD:
        print(f"⚠️  Я НЕ УВЕРЕНА")
        print(f"   Модель колеблется между классами:")
        print(f"      {CLASSES[sorted_idx[0].item()]}: {top1_prob*100:.2f}%")
        print(f"      {CLASSES[sorted_idx[1].item()]}: {top2_prob*100:.2f}%")
        print(f"   Разрыв слишком маленький ({margin*100:.1f}%).")
        print(f"   Рекомендуется консультация специалиста.")
    else:
        print(f"🔬 Диагноз: {predicted_class}")
        print(f"📊 Уверенность: {top1_prob*100:.2f}%")
        print(f"📏 Разрыв с 2-м классом: {margin*100:.2f}%")

    # Распределение вероятностей (всегда показываем)
    print(f"\n{'-'*60}")
    print("Распределение вероятностей:")
    for i in range(len(CLASSES)):
        idx = sorted_idx[i].item()
        prob = sorted_probs[i].item()
        bar = '█' * int(prob * 30)
        print(f"  {CLASSES[idx]:<25} {prob*100:6.2f}%  {bar}")
    print(f"{'='*60}\n")


# --- Точка входа ---
if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        predict_image(r"E:\3 курс\Lor_Ai\test_Photos\Наружный отит\images (1).jpg")
        predict_image(r"E:\3 курс\Lor_Ai\test_Photos\Наружный отит\images.jpg")
        predict_image(r"E:\3 курс\Lor_Ai\test_Photos\Холестеатома\images.jpg")
        predict_image(r"E:\3 курс\Lor_Ai\test_Photos\норма\images (1).jpg")