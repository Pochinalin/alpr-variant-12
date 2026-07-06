# src/visualization.py
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

def draw_mask_overlay(image, mask, alpha=0.5):
    """
    Наложение маски на изображение
    """
    if mask is None or mask.size == 0:
        return image
    
    # Конвертация в numpy если нужно
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Убеждаемся что маска в правильном формате
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Создаем цветную маску
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask * 255  # Красный канал
    
    # Наложение маски
    overlay = image.copy()
    mask_alpha = (mask > 0.5).astype(np.uint8)
    
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_alpha > 0,
            image[:, :, c] * (1 - alpha) + mask_colored[:, :, c] * alpha,
            image[:, :, c]
        )
    
    return overlay

def create_visualization(image, bbox, mask=None, text="", confidence=0):
    """
    Создание полной визуализации с маской и bbox
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Рисуем bbox
    if bbox and len(bbox) == 4 and bbox != [0, 0, 0, 0]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Добавляем текст
        if text:
            text_display = f"{text}"
            (text_w, text_h), _ = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 15), (x1 + text_w + 15, y1), (0, 0, 0), -1)
            cv2.putText(img, text_display, (x1 + 8, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Добавляем уверенность
        conf_text = f"Conf: {confidence:.2f}"
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y2 + 5), (x1 + conf_w + 10, y2 + conf_h + 10), (0, 0, 0), -1)
        cv2.putText(img, conf_text, (x1 + 5, y2 + conf_h + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Наложение маски
    if mask is not None and mask.size > 0:
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Создаем полупрозрачную маску
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_alpha = (mask_resized > 0.5).astype(np.uint8)
        
        # Красная маска
        overlay = img.copy()
        overlay[:, :, 0] = np.where(mask_alpha > 0, 255, overlay[:, :, 0])
        overlay[:, :, 1] = np.where(mask_alpha > 0, overlay[:, :, 1] * 0.5, overlay[:, :, 1])
        overlay[:, :, 2] = np.where(mask_alpha > 0, overlay[:, :, 2] * 0.5, overlay[:, :, 2])
        
        # Смешивание
        alpha = 0.4
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    return img

def create_side_by_side(image, mask=None, bbox=None, text="", confidence=0):
    """
    Создание сравнения исходного изображения и результата
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Создаем результат
    result = create_visualization(img, bbox, mask, text, confidence)
    
    # Объединяем изображения
    h, w = img.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = img
    combined[:, w:] = result
    
    # Добавляем подписи
    cv2.putText(combined, "Исходное", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Результат", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined

def create_mask_only(mask, size=(400, 400)):
    """
    Визуализация только маски
    """
    if mask is None or mask.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Нормализация
    mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    mask_normalized = (mask_normalized * 255).astype(np.uint8)
    
    # Изменение размера
    mask_resized = cv2.resize(mask_normalized, size)
    
    # Цветная маска
    mask_color = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    mask_color[:, :, 0] = mask_resized  # Красный канал
    mask_color[:, :, 1] = mask_resized * 0.3  # Немного зеленого
    mask_color[:, :, 2] = 0
    
    return mask_color