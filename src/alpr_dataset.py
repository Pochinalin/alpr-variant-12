# src/alpr_dataset.py
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
import random


class PlateSegmentationDataset(Dataset):
    """
    Датасет для сегментации номерных знаков
    Формат: путь_к_изображению x1 y1 x2 y2 x3 y3 x4 y4
    """
    
    def __init__(self, file_list, size=64, augment=True, use_green_channel=True):
        self.file_list = Path(file_list)
        self.size = size
        self.augment = augment
        self.use_green_channel = use_green_channel
        self.samples = []
        
        # Загрузка данных
        with open(self.file_list, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(line)
        
        print(f"Загружено {len(self.samples)} образцов из {self.file_list}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        parts = sample.split()
        
        if len(parts) < 1:
            return self.__getitem__((idx + 1) % len(self.samples))
        
        img_path = Path(parts[0])
        
        # Загрузка изображения
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Не удалось загрузить: {img_path}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Всегда конвертируем в оттенки серого (1 канал)
        if self.use_green_channel:
            # Используем зеленый канал (лучше для номеров)
            img = img[:, :, 1]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Создание маски
        mask = np.zeros_like(img, dtype=np.float32)
        
        # Парсинг полигона
        if len(parts) > 1:
            points = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        points.append([x, y])
                    except ValueError:
                        continue
            
            if len(points) >= 3:
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1.0)
        
        # Изменение размера
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        
        # Нормализация
        img = img.astype(np.float32) / 255.0
        
        # Аугментация
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # Добавление размерности канала (1 канал)
        img = np.expand_dims(img, axis=0)   # [1, size, size]
        mask = np.expand_dims(mask, axis=0) # [1, size, size]
        
        return torch.FloatTensor(img), torch.FloatTensor(mask)
    
    def _augment(self, img, mask):
        """Аугментация данных"""
        # Горизонтальное отражение
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        # Поворот
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            from scipy.ndimage import rotate
            img = rotate(img, angle, reshape=False, order=1)
            mask = rotate(mask, angle, reshape=False, order=0)
        
        # Сдвиг
        if random.random() > 0.8:
            shift_x = random.uniform(-5, 5)
            shift_y = random.uniform(-5, 5)
            from scipy.ndimage import shift
            img = shift(img, (shift_y, shift_x), order=1)
            mask = shift(mask, (shift_y, shift_x), order=0)
        
        # Яркость
        if random.random() > 0.7:
            brightness = random.uniform(0.7, 1.3)
            img = img * brightness
            img = np.clip(img, 0, 1)
        
        # Контраст
        if random.random() > 0.7:
            contrast = random.uniform(0.7, 1.3)
            mean = img.mean()
            img = (img - mean) * contrast + mean
            img = np.clip(img, 0, 1)
        
        return img, mask