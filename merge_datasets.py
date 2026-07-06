# merge_datasets.py
import os
import shutil
import random
from pathlib import Path
import yaml
import cv2
import numpy as np

def merge_yolo_datasets(dataset1_path, dataset2_path, output_path="data"):
    """
    Объединение двух YOLO датасетов
    """
    print("="*60)
    print("ОБЪЕДИНЕНИЕ ДАТАСЕТОВ")
    print("="*60)
    
    dataset1 = Path(dataset1_path)
    dataset2 = Path(dataset2_path)
    output = Path(output_path)
    
    # Создание структуры папок
    for split in ['train', 'val', 'test']:
        (output / split / 'images').mkdir(parents=True, exist_ok=True)
        (output / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_labels = 0
    
    # Функция для копирования данных из одного датасета
    def copy_dataset(source, dest, split_name):
        nonlocal total_images, total_labels
        
        source_img_dir = source / split_name / 'images'
        source_label_dir = source / split_name / 'labels'
        
        if not source_img_dir.exists():
            print(f"⚠️ Папка не найдена: {source_img_dir}")
            return
        
        dest_img_dir = dest / split_name / 'images'
        dest_label_dir = dest / split_name / 'labels'
        
        # Копирование изображений
        for img_path in source_img_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                dest_path = dest_img_dir / img_path.name
                
                # Если файл уже существует, добавляем суффикс
                if dest_path.exists():
                    name = img_path.stem + '_copy'
                    dest_path = dest_img_dir / f"{name}{img_path.suffix}"
                
                shutil.copy2(img_path, dest_path)
                total_images += 1
                
                # Копирование соответствующей разметки
                label_path = source_label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    dest_label = dest_label_dir / f"{dest_path.stem}.txt"
                    shutil.copy2(label_path, dest_label)
                    total_labels += 1
                else:
                    print(f"  ⚠️ Разметка не найдена: {label_path}")
        
        print(f"  ✅ {split_name}: скопировано изображений: {len(list(source_img_dir.glob('*.*')))}")
    
    print("\n1. Копирование Dataset 1...")
    copy_dataset(dataset1, output, 'train')
    copy_dataset(dataset1, output, 'val')
    copy_dataset(dataset1, output, 'test')
    
    print("\n2. Копирование Dataset 2...")
    copy_dataset(dataset2, output, 'train')
    copy_dataset(dataset2, output, 'val')
    copy_dataset(dataset2, output, 'test')
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОБЪЕДИНЕНИЯ")
    print("="*60)
    print(f"Всего изображений: {total_images}")
    print(f"Всего файлов разметки: {total_labels}")
    
    # Создание data.yaml
    create_data_yaml(output, total_images, total_labels)
    
    # Создание файлов разметки для ALPR
    create_alpr_splits(output)
    
    return output

def create_data_yaml(data_dir, total_images, total_labels):
    """Создание data.yaml для YOLO"""
    data_dir = Path(data_dir)
    
    yaml_content = f"""
path: {str(data_dir.absolute())}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['number_plate']

# Статистика датасета
total_images: {total_images}
total_labels: {total_labels}
"""
    
    yaml_path = data_dir / 'data.yaml'
    yaml_path.write_text(yaml_content, encoding='utf-8')
    print(f"\n✅ Создан {yaml_path}")

def create_alpr_splits(data_dir):
    """Создание файлов разметки для ALPR из YOLO формата"""
    data_dir = Path(data_dir)
    splits_dir = data_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        img_dir = data_dir / split / 'images'
        label_dir = data_dir / split / 'labels'
        output_file = splits_dir / f'{split}.txt'
        
        output_file.write_text('')
        
        if not img_dir.exists():
            print(f"⚠️ Папка не найдена: {img_dir}")
            continue
        
        for img_path in img_dir.glob('*.jpg'):
            # Загрузка изображения для получения размеров
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Поиск разметки
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                if class_id != 0:  # Только номерные знаки
                    continue
                
                # Конвертация YOLO в ALPR формат
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h
                
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center - height/2)
                x3 = int(x_center + width/2)
                y3 = int(y_center + height/2)
                x4 = int(x_center - width/2)
                y4 = int(y_center + height/2)
                
                # Запись в ALPR формат
                with open(output_file, 'a') as f:
                    f.write(f"{img_path} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
        
        print(f"  ✅ {split}.txt: {sum(1 for _ in open(output_file))} записей")
    
    print(f"\n✅ ALPR разметка создана в {splits_dir}")

def prepare_for_training():
    """Основная функция подготовки данных"""
    print("="*60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("="*60)
    
    # Пути к датасетам (укажите свои пути)
    # Путь к первому датасету (car-plate-russian)
    dataset1_path = Path("datasets/car-plate-russian-1")
    
    # Путь ко второму датасету (number-ozskg)
    dataset2_path = Path("datasets/number-ozskg-1")
    
    # Проверяем наличие датасетов
    if not dataset1_path.exists() or not dataset2_path.exists():
        print("\n❌ Датасеты не найдены!")
        print("\nПожалуйста, укажите правильные пути:")
        print(f"  Датасет 1 (car-plate-russian): {dataset1_path}")
        print(f"  Датасет 2 (number-ozskg): {dataset2_path}")
        print("\nИли измените пути в скрипте")
        
        # Предлагаем ввести пути вручную
        d1 = input("\nВведите путь к первому датасету (car-plate-russian): ").strip()
        if d1:
            dataset1_path = Path(d1)
        
        d2 = input("Введите путь ко второму датасету (number-ozskg): ").strip()
        if d2:
            dataset2_path = Path(d2)
    
    if not dataset1_path.exists():
        print(f"❌ Датасет 1 не найден: {dataset1_path}")
        return
    
    if not dataset2_path.exists():
        print(f"❌ Датасет 2 не найден: {dataset2_path}")
        return
    
    print(f"\n✅ Найден датасет 1: {dataset1_path}")
    print(f"✅ Найден датасет 2: {dataset2_path}")
    
    # Объединение датасетов
    output_path = Path("data")
    merge_yolo_datasets(dataset1_path, dataset2_path, output_path)
    
    print("\n" + "="*60)
    print("✅ ДАННЫЕ ГОТОВЫ ДЛЯ ОБУЧЕНИЯ!")
    print("="*60)
    print("\nЗапустите 02_TRAIN_MODELS.cmd для начала обучения")

def check_dataset_structure():
    """Проверка структуры датасета"""
    dataset_paths = [
        Path("datasets/car-plate-russian-1"),
        Path("datasets/number-ozskg-1"),
        Path("data")
    ]
    
    print("="*60)
    print("ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТОВ")
    print("="*60)
    
    for path in dataset_paths:
        if not path.exists():
            print(f"\n❌ {path} не найден")
            continue
        
        print(f"\n✅ {path}")
        
        for split in ['train', 'val', 'test']:
            img_dir = path / split / 'images'
            label_dir = path / split / 'labels'
            
            if img_dir.exists():
                count = len(list(img_dir.glob('*.*')))
                print(f"  {split}: {count} изображений")
            
            if label_dir.exists():
                count = len(list(label_dir.glob('*.txt')))
                print(f"  {split}: {count} файлов разметки")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_dataset_structure()
    else:
        prepare_for_training()