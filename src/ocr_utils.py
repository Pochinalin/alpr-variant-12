# src/ocr_utils.py
import cv2
import numpy as np
import subprocess
import sys

def ocr_runtime_status():
    """Проверка статуса OCR"""
    statuses = []
    tesseract_ok = False
    rapid_ok = False
    
    # Проверка Tesseract
    try:
        import pytesseract
        result = subprocess.run(['tesseract', '--version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            statuses.append("Tesseract: OK")
            tesseract_ok = True
        else:
            statuses.append("Tesseract: не найден в PATH")
    except:
        statuses.append("Tesseract: не установлен")
    
    # Проверка RapidOCR
    try:
        import rapidocr_onnxruntime
        ocr = rapidocr_onnxruntime.RapidOCR()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ocr(test_img)
        statuses.append("RapidOCR: OK")
        rapid_ok = True
    except:
        statuses.append("RapidOCR: не установлен")
    
    status_msg = "; ".join(statuses)
    return tesseract_ok or rapid_ok, status_msg

def enhance_for_ocr(image):
    """Улучшение изображения для OCR"""
    try:
        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Увеличение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Бинаризация
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Удаление шума
        denoised = cv2.medianBlur(binary, 3)
        
        # Увеличение размера
        if denoised.shape[0] < 100 or denoised.shape[1] < 200:
            scale = max(2, int(300 / denoised.shape[1]))
            denoised = cv2.resize(denoised, (denoised.shape[1] * scale, denoised.shape[0] * scale))
        
        return denoised
    except Exception as e:
        return image

def recognize_text(image):
    """Распознавание текста с использованием Tesseract"""
    
    # 1. Сначала пробуем Tesseract
    try:
        import pytesseract
        
        # Проверяем, что Tesseract доступен
        subprocess.run(['tesseract', '--version'], 
                      capture_output=True, text=True, timeout=5, check=True)
        
        # Улучшение изображения
        enhanced = enhance_for_ocr(image)
        
        # Настройка для распознавания номеров
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АВЕКМНОРСТУХ'
        
        # Распознавание
        text = pytesseract.image_to_string(enhanced, config=custom_config, lang='rus+eng')
        text = text.strip().replace(' ', '').replace('\n', '').replace('\r', '')
        
        if text:
            return text, 0.85, "tesseract_ok"
        
        # Пробуем с оригинальным изображением
        text = pytesseract.image_to_string(image, config=custom_config, lang='rus+eng')
        text = text.strip().replace(' ', '').replace('\n', '').replace('\r', '')
        
        if text:
            return text, 0.75, "tesseract_ok"
            
    except Exception as e:
        pass  # Переходим к следующему методу
    
    # 2. Пробуем RapidOCR
    try:
        import rapidocr_onnxruntime
        
        enhanced = enhance_for_ocr(image)
        ocr = rapidocr_onnxruntime.RapidOCR()
        result = ocr(enhanced)
        
        if result and len(result) > 0 and len(result[0]) > 0:
            texts = []
            for box in result[0]:
                if len(box) >= 2:
                    text = box[1][0] if isinstance(box[1], (list, tuple)) else str(box[1])
                    texts.append(text)
            
            if texts:
                full_text = ''.join(texts).strip().replace(' ', '').replace('\n', '')
                if full_text:
                    return full_text, 0.7, "rapidocr_ok"
                    
    except:
        pass
    
    return "", 0.0, "no_text_recognized"