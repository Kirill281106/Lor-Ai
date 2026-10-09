import io

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Модель, трансформации, классы и пороги берём из predict.py без изменений
from predict import CLASSES, CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD, model, transform

app = FastAPI(title="Lor-Ai API")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при открытии: {e}")

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

    # 🎯 Та же логика определения "не знаю", что и в predict.py
    if top1_prob < CONFIDENCE_THRESHOLD:
        status = "unknown"
        message = "Я НЕ ЗНАЮ, ЧТО ЭТО"
    elif margin < MARGIN_THRESHOLD:
        status = "uncertain"
        message = "Я НЕ УВЕРЕНА"
    else:
        status = "diagnosis"
        message = f"Диагноз: {predicted_class}"

    return {
        "file": file.filename,
        "status": status,
        "message": message,
        "diagnosis": predicted_class if status == "diagnosis" else None,
        "confidence": round(top1_prob * 100, 2),
        "margin": round(margin * 100, 2),
        "probabilities": {
            CLASSES[sorted_idx[i].item()]: round(sorted_probs[i].item() * 100, 2)
            for i in range(len(CLASSES))
        },
    }
