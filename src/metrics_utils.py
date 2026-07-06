# src/metrics_utils.py
import torch
import numpy as np


def count_parameters(model):
    """
    Подсчет количества параметров модели
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_confusion_matrix(pred, target, threshold=0.5):
    """
    Расчет матрицы ошибок для бинарной сегментации
    """
    pred_binary = (pred > threshold).float()
    
    tp = ((pred_binary == 1) & (target == 1)).sum().item()
    fp = ((pred_binary == 1) & (target == 0)).sum().item()
    tn = ((pred_binary == 0) & (target == 0)).sum().item()
    fn = ((pred_binary == 0) & (target == 1)).sum().item()
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }


def calculate_metrics(confusion_matrix):
    """
    Расчет метрик из матрицы ошибок
    """
    tp = confusion_matrix['TP']
    fp = confusion_matrix['FP']
    tn = confusion_matrix['TN']
    fn = confusion_matrix['FN']
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }