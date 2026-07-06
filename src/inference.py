# src/inference.py
import cv2
import numpy as np
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import create_model
from src.ocr_utils import recognize_text

def load_model_weights(model, checkpoint_path, device='cpu'):
    """
    Загрузка весов модели с поддержкой разных форматов
    """
    if not checkpoint_path.exists():
        print(f"⚠️ Файл модели не найден: {checkpoint_path}")
        return model
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'], strict=False)
                print(f"✅ Загружены веса модели (эпоха: {checkpoint.get('epoch', 'unknown')})")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("✅ Загружены веса модели (state_dict)")
            else:
                try:
                    model.load_state_dict(checkpoint, strict=False)
                    print("✅ Загружены веса модели")
                except:
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint.items() 
                                     if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict, strict=False)
                    print(f"✅ Загружены частичные веса ({len(pretrained_dict)}/{len(model_dict)})")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("✅ Загружены веса модели")
            
    except Exception as e:
        print(f"⚠️ Ошибка загрузки модели: {e}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'], strict=False)
                print("✅ Загружены веса модели (strict=False)")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("✅ Загружены веса модели (strict=False)")
        except Exception as e2:
            print(f"❌ Не удалось загрузить модель: {e2}")
    
    return model

def preprocess_image_rgb(image, size=64):
    """
    Предобработка изображения для RGB модели (3 канала)
    """
    if len(image.shape) == 2:
        # Если изображение в оттенках серого, конвертируем в RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # Если RGBA, конвертируем в RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Изменение размера
    resized = cv2.resize(image, (size, size))
    
    # Нормализация [0, 255] -> [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Изменение порядка каналов (HWC -> CHW)
    normalized = np.transpose(normalized, (2, 0, 1))
    
    # Преобразование в тензор
    tensor = torch.FloatTensor(normalized).unsqueeze(0)  # [1, 3, size, size]
    
    return tensor, resized

def postprocess_mask(mask, original_shape, threshold=0.5):
    """Постобработка маски"""
    mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))
    binary_mask = (mask_resized > threshold).astype(np.uint8)
    return binary_mask

def find_plate_bbox(mask, original_image):
    """Нахождение bounding box номера по маске"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0, 0, 0, 0], 0.0, None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    aspect_ratio = w / h if h > 0 else 0
    confidence = 0.9 if 1.5 < aspect_ratio < 5 else 0.6
    
    padding = 15
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(original_image.shape[1], x + w + padding)
    y2 = min(original_image.shape[0], y + h + padding)
    
    plate_crop = original_image[y1:y2, x1:x2]
    
    return [int(x1), int(y1), int(x2), int(y2)], confidence, plate_crop

def enhance_plate(plate_img):
    """Улучшение изображения номера для OCR"""
    if plate_img is None or plate_img.size == 0:
        return None
    
    # Конвертация в оттенки серого для OCR
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = plate_img
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, 3)
    
    if denoised.shape[0] < 100 or denoised.shape[1] < 300:
        scale_x = max(2, int(400 / denoised.shape[1]))
        scale_y = max(2, int(150 / denoised.shape[0]))
        denoised = cv2.resize(denoised, (denoised.shape[1] * scale_x, denoised.shape[0] * scale_y))
    
    return denoised

def run_alpr(image, model_name, checkpoint_path, threshold=0.5, device='cpu'):
    """Запуск ALPR на изображении"""
    
    # Создание модели с 3 входными каналами (RGB)
    model = create_model(model_name, in_channels=3)
    
    # Загрузка весов
    model = load_model_weights(model, checkpoint_path, device)
    model.to(device)
    model.eval()
    
    # Предобработка для RGB
    tensor, resized = preprocess_image_rgb(image, size=64)
    tensor = tensor.to(device)
    
    # Инференс
    with torch.no_grad():
        output = model(tensor)
        mask = output.cpu().numpy()[0, 0]
    
    # Постобработка
    binary_mask = postprocess_mask(mask, image.shape[:2], threshold)
    
    # Нахождение bounding box
    bbox, confidence, crop = find_plate_bbox(binary_mask, image)
    
    # Распознавание текста
    ocr_text = ""
    ocr_confidence = 0.0
    ocr_status = "no_text"
    
    if crop is not None and crop.size > 0:
        enhanced = enhance_plate(crop)
        if enhanced is not None:
            ocr_text, ocr_confidence, ocr_status = recognize_text(enhanced)
        if not ocr_text:
            ocr_text, ocr_confidence, ocr_status = recognize_text(crop)
    
    # Создание overlay
    overlay = image.copy()
    if bbox != [0, 0, 0, 0]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if ocr_text:
            text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)
            cv2.putText(overlay, ocr_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    area_percent = (binary_mask.sum() / (image.shape[0] * image.shape[1])) * 100
    
    result = {
        'bbox': bbox,
        'confidence': float(confidence),
        'area_percent': float(area_percent),
        'text': ocr_text,
        'ocr_confidence': float(ocr_confidence),
        'ocr_engine': ocr_status,
        'crop': crop,
        'overlay': overlay,
        'detection_method': f'{model_name}_segmentation',
        'candidate_count': 1
    }
    
    return result