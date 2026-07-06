# train.py
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import MODEL_NAMES, create_model, model_display_names
from src.alpr_dataset import PlateSegmentationDataset
from src.training_utils import train_one_model
from src.metrics_utils import count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_subset(dataset, limit: int, seed: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:limit])


def create_summary_csv(summaries: list[dict], output_path: Path) -> None:
    """Создание CSV файла с результатами обучения"""
    if not summaries:
        return
    
    # Определяем поля для CSV
    fieldnames = [
        'model',
        'display_name',
        'parameters',
        'epochs_requested',
        'epochs_finished',
        'best_epoch',
        'best_val_dice',
        'best_val_iou',
        'batch_size',
        'input_size',
        'train_samples',
        'val_samples',
        'duration_seconds'
    ]
    
    # Загружаем существующие данные
    existing: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['model']] = row
    
    # Обновляем данными
    for row in summaries:
        # Берем только нужные поля
        clean_row = {k: row.get(k, '') for k in fieldnames}
        existing[row['model']] = clean_row
    
    # Сохраняем в правильном порядке
    ordered = [existing[name] for name in MODEL_NAMES if name in existing]
    if ordered:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
        print(f"✅ Сохранено: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Training five different architectures for plate localization.')
    parser.add_argument('--model', default='all', choices=['all'] + MODEL_NAMES)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-train', type=int, default=0)
    parser.add_argument('--max-val', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32:
        parser.error('Invalid training parameters')
    
    train_list = ROOT / 'data/splits/train.txt'
    val_list = ROOT / 'data/splits/val.txt'
    
    if not train_list.is_file() or not val_list.is_file():
        print(f"[ОШИБКА] Файлы разметки не найдены:")
        print(f"  {train_list}")
        print(f"  {val_list}")
        print("\nЗапустите prepare_dataset.py для подготовки данных")
        sys.exit(1)
    
    torch.set_num_threads(max(1, args.threads))
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("ОБУЧЕНИЕ 5 РАЗНЫХ АРХИТЕКТУР")
    print("="*60)
    print(f"Устройство: {device}")
    print(f"Эпох: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Размер входа: {args.size}x{args.size}")
    print("\nАрхитектуры:")
    for name in MODEL_NAMES:
        print(f"  • {model_display_names()[name]}")
    print("="*60)
    
    print("\nЗагрузка данных...")
    train_ds = PlateSegmentationDataset(train_list, size=args.size, augment=True)
    val_ds = PlateSegmentationDataset(val_list, size=args.size, augment=False)
    
    train_ds = limited_subset(train_ds, args.max_train, args.seed)
    val_ds = limited_subset(val_ds, args.max_val, args.seed + 1)
    
    print(f"Тренировочных: {len(train_ds)}")
    print(f"Валидационных: {len(val_ds)}")
    
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    selected = MODEL_NAMES if args.model == 'all' else [args.model]
    summaries = []
    start_total = time.time()
    
    for idx, name in enumerate(selected, 1):
        print("\n" + "="*60)
        print(f"МОДЕЛЬ {idx}/{len(selected)}: {model_display_names()[name]}")
        print("="*60)
        
        set_seed(args.seed + idx)
        
        try:
            model = create_model(name, in_channels=1)
            params = count_parameters(model)
            print(f"Параметров: {params:,}")
            
            out_dir = ROOT / 'checkpoints' / name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            history, best_metrics = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                out_dir=out_dir,
                model_name=name,
                input_size=args.size,
            )
            duration = time.time() - start_time
            
            best = max(history, key=lambda row: row.get('Dice/F1', 0))
            summary = {
                'model': name,
                'display_name': model_display_names()[name],
                'parameters': params,
                'epochs_requested': args.epochs,
                'epochs_finished': len(history),
                'best_epoch': best.get('epoch', 0),
                'best_val_dice': best.get('Dice/F1', 0),
                'best_val_iou': best.get('IoU', 0),
                'batch_size': args.batch_size,
                'input_size': args.size,
                'train_samples': len(train_ds),
                'val_samples': len(val_ds),
                'duration_seconds': round(duration, 2),
                'learning_rate': args.lr,
                'seed': args.seed,
            }
            
            # Сохраняем summary.json
            with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            summaries.append(summary)
            
            print(f"\n✅ {model_display_names()[name]} обучена!")
            print(f"   Dice/F1: {summary['best_val_dice']:.4f}")
            print(f"   IoU: {summary['best_val_iou']:.4f}")
            print(f"   Время: {duration:.1f} сек")
            
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сохранение результатов
    if summaries:
        summary_path = ROOT / 'checkpoints/training_summary.csv'
        create_summary_csv(summaries, summary_path)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("="*60)
        sorted_summaries = sorted(summaries, key=lambda x: x['best_val_dice'], reverse=True)
        for i, s in enumerate(sorted_summaries, 1):
            print(f"{i}. {s['display_name']}: Dice={s['best_val_dice']:.4f}, IoU={s['best_val_iou']:.4f}")
    
    print(f"\nОбщее время: {time.time() - start_total:.1f} сек")

if __name__ == '__main__':
    main()