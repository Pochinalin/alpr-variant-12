# src/training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json
import numpy as np


def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_dice = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Убеждаемся, что размеры совпадают
        if outputs.shape != masks.shape:
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[2:], mode='bilinear', align_corners=True
            )
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        dice = 1 - dice_loss(outputs, masks).item()
        total_dice += dice
    
    return total_loss / len(loader), total_dice / len(loader)


def validate(model, loader, device):
    model.eval()
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            if outputs.shape != masks.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[2:], mode='bilinear', align_corners=True
                )
            
            dice = 1 - dice_loss(outputs, masks).item()
            iou = iou_score(outputs, masks)
            
            total_dice += dice
            total_iou += iou
    
    return total_dice / len(loader), total_iou / len(loader)


def train_one_model(model, train_loader, val_loader, epochs, lr, device, 
                   out_dir, model_name, input_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = []
    best_dice = 0
    best_metrics = {}
    start_time = time.time()
    
    print(f"Начало обучения {model_name}...")
    print(f"Устройство: {device}, LR: {lr}, Эпох: {epochs}")
    
    for epoch in range(epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_dice, val_iou = validate(model, val_loader, device)
        
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'Dice/F1': val_dice,
            'IoU': val_iou
        }
        history.append(epoch_data)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={train_loss:.4f}, "
                  f"Train Dice={train_dice:.4f}, "
                  f"Val Dice={val_dice:.4f}, "
                  f"Val IoU={val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            best_metrics = {
                'epoch': epoch + 1,
                'dice': val_dice,
                'iou': val_iou,
                'train_loss': train_loss
            }
            torch.save(model.state_dict(), out_dir / 'best.pt')
            with open(out_dir / 'best_metrics.json', 'w') as f:
                json.dump(best_metrics, f, indent=2)
        
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), out_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    duration = time.time() - start_time
    
    print(f"\nОбучение {model_name} завершено!")
    print(f"Лучший Dice/F1: {best_dice:.4f} (эпоха {best_metrics.get('epoch', '?')})")
    print(f"Время: {duration:.1f} секунд")
    
    return history, best_metrics