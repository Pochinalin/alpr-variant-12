from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import run_alpr
from src.ocr_utils import ocr_runtime_status
from src.models import MODEL_NAMES, model_display_names

LOG_PATH = ROOT / 'logs/predictions.jsonl'
COMPARISON_LOG = ROOT / 'logs/model_comparison.jsonl'

# ===================== НАСТРОЙКА СТРАНИЦЫ =====================

st.set_page_config(
    page_title='ALPR - Сравнение 5 архитектур',
    page_icon='🚘',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ===================== ТЕМНАЯ ТЕМА =====================

st.markdown("""
<style>
    /* Основной фон */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Заголовки */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2ffc, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
        text-shadow: 0 0 40px rgba(0, 212, 255, 0.3);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #1d2d50;
        padding-bottom: 0.5rem;
    }
    
    /* Карточки */
    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #0e1117);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2a2f3f;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00d4ff;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7b2ffc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 0.3rem;
    }
    
    /* Боксы */
    .best-model-box {
        background: linear-gradient(145deg, #0a1f1a, #0e1117);
        border-left: 4px solid #00d4ff;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #1a2f2a;
    }
    
    .error-box {
        background: linear-gradient(145deg, #1f0a0a, #0e1117);
        border-left: 4px solid #ff6b6b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #2f1a1a;
    }
    
    .info-box {
        background: linear-gradient(145deg, #0a1a2f, #0e1117);
        border-left: 4px solid #4dabf7;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #1a2a3f;
    }
    
    .success-box {
        background: linear-gradient(145deg, #0a1f0a, #0e1117);
        border-left: 4px solid #51cf66;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #1a2f1a;
    }
    
    /* Текст номера */
    .plate-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        font-family: 'Courier New', monospace;
        letter-spacing: 4px;
        background: linear-gradient(145deg, #0a1a1a, #0e1117);
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        display: inline-block;
        border: 1px solid #1a2a2a;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
    }
    
    /* Таблицы */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        background: #0e1117;
    }
    .stDataFrame thead th {
        background: #1a1f2e !important;
        color: #00d4ff !important;
        font-weight: 600;
    }
    .stDataFrame tbody td {
        color: #e0e0e0 !important;
    }
    .stDataFrame tbody tr:hover {
        background: #1a1f2e !important;
    }
    
    /* Боковая панель */
    .css-1d391kg {
        background-color: #0a0d13 !important;
    }
    
    /* Кнопки */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff, #7b2ffc) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #1a1f2e !important;
        border: 2px dashed #2a2f3f !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    .stFileUploader:hover {
        border-color: #00d4ff !important;
    }
    
    /* Изображения */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2a2f3f;
    }
    
    /* Скроллбар */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #7b2ffc);
        border-radius: 4px;
    }
    
    /* Divider */
    hr {
        border-color: #1d2d50 !important;
        margin: 1.5rem 0 !important;
    }
    
    .stCaption {
        color: #8892b0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================== ФУНКЦИИ =====================

def available_models() -> list[str]:
    models = []
    for name in MODEL_NAMES:
        if (ROOT / 'checkpoints' / name / 'best.pt').exists() or \
           (ROOT / 'checkpoints' / name / 'best_weights.pt').exists():
            models.append(name)
    return models if models else MODEL_NAMES

def get_model_metrics(model_name: str) -> dict:
    summary_path = ROOT / 'checkpoints' / model_name / 'summary.json'
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def default_model(models: list[str]) -> str:
    best_name = models[0]
    best_score = -1.0
    for name in models:
        metrics = get_model_metrics(name)
        score = metrics.get('best_val_dice', -1.0)
        if score > best_score:
            best_score, best_name = score, name
    return best_name

def read_comparison_history() -> pd.DataFrame:
    if not COMPARISON_LOG.exists():
        return pd.DataFrame()
    rows = []
    for line in COMPARISON_LOG.read_text(encoding='utf-8').splitlines():
        try:
            data = json.loads(line)
            if 'models' in data:
                for model_name, model_data in data['models'].items():
                    rows.append({
                        'timestamp': data.get('timestamp', ''),
                        'image': data.get('image', ''),
                        'threshold': data.get('threshold', 0.5),
                        'model': model_name,
                        'confidence': model_data.get('confidence', 0),
                        'time_ms': model_data.get('time_ms', 0),
                        'text': model_data.get('text', ''),
                        'ocr_confidence': model_data.get('ocr_confidence', 0),
                        'area_percent': model_data.get('area_percent', 0),
                        'bbox': model_data.get('bbox', [])
                    })
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)

def append_comparison_log(comparison_data: dict) -> None:
    COMPARISON_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')

def process_with_all_models(image, models_to_test, threshold=0.5, device='cpu'):
    results = {}
    times = {}
    errors = {}
    
    for model_name in models_to_test:
        checkpoint_path = ROOT / 'checkpoints' / model_name / 'best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = ROOT / 'checkpoints' / model_name / 'best_weights.pt'
        
        if not checkpoint_path.exists():
            results[model_name] = {'error': f'Файл модели не найден'}
            times[model_name] = 0
            errors[model_name] = 'Модель не найдена'
            continue
        
        start_time = time.perf_counter()
        try:
            result = run_alpr(
                np.array(image),
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
                device=device,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            results[model_name] = result
            times[model_name] = elapsed_ms
            errors[model_name] = None
        except Exception as e:
            results[model_name] = {'error': str(e)}
            times[model_name] = 0
            errors[model_name] = str(e)
    
    return results, times, errors

def create_comparison_metrics(results, times, errors):
    data = []
    for model_name, result in results.items():
        display_name = model_display_names().get(model_name, model_name)
        
        if 'error' in result:
            data.append({
                'Модель': display_name,
                'Уверенность': 0,
                'Время (мс)': times.get(model_name, 0),
                'Текст': 'Ошибка',
                'Площадь маски (%)': 0,
                'OCR уверенность': 0,
                'Статус': '❌ Ошибка',
                'Ошибка': result.get('error', 'Неизвестная ошибка'),
                'bbox': []
            })
        else:
            data.append({
                'Модель': display_name,
                'Уверенность': result.get('confidence', 0),
                'Время (мс)': times.get(model_name, 0),
                'Текст': result.get('text', 'Не распознан'),
                'Площадь маски (%)': result.get('area_percent', 0),
                'OCR уверенность': result.get('ocr_confidence', 0),
                'Статус': '✅ Распознан' if result.get('text') else '⚠️ Нет текста',
                'Ошибка': '',
                'bbox': result.get('bbox', [])
            })
    return pd.DataFrame(data)

def create_model_statistics():
    history = read_comparison_history()
    if history.empty:
        return None
    
    stats = []
    for model_name in MODEL_NAMES:
        model_data = history[history['model'] == model_name]
        if model_data.empty:
            continue
        
        stats.append({
            'Модель': model_display_names().get(model_name, model_name),
            'Средняя уверенность': model_data['confidence'].mean(),
            'Макс уверенность': model_data['confidence'].max(),
            'Среднее время (мс)': model_data['time_ms'].mean(),
            'Кол-во обработанных': len(model_data),
            'Успешных распознаваний': (model_data['text'] != '').sum(),
            'Точность': (model_data['text'] != '').sum() / len(model_data) if len(model_data) > 0 else 0
        })
    
    return pd.DataFrame(stats)

def draw_bbox_on_image(image, bbox, text, confidence):
    """Рисование bbox на изображении без маски"""
    img_copy = np.array(image).copy()
    
    if bbox and len(bbox) == 4 and bbox != [0, 0, 0, 0]:
        x1, y1, x2, y2 = bbox
        
        # Рисуем рамку
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Рисуем фон для текста
        if text:
            text_display = f"{text}"
            (text_w, text_h), _ = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
            cv2.putText(img_copy, text_display, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Добавляем уверенность
        conf_text = f"Conf: {confidence:.2f}"
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y2 + 5), (x1 + conf_w + 10, y2 + conf_h + 10), (0, 0, 0), -1)
        cv2.putText(img_copy, conf_text, (x1 + 5, y2 + conf_h + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img_copy

def create_comparison_chart(comparison_df):
    """Создание графиков сравнения с темной темой"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '🎯 Уверенность локализации',
            '⏱️ Время обработки (мс)',
            '📐 Площадь маски (%)',
            '📝 OCR уверенность'
        )
    )
    
    models = comparison_df['Модель'].tolist()
    
    colors = {
        'primary': '#00d4ff',
        'secondary': '#7b2ffc',
        'success': '#51cf66',
        'warning': '#ffd43b',
        'danger': '#ff6b6b',
        'default': '#6c7a89'
    }
    
    # Уверенность
    fig.add_trace(
        go.Bar(name='Уверенность', x=models, y=comparison_df['Уверенность'],
               text=comparison_df['Уверенность'].round(3), textposition='auto',
               textfont_color='#ffffff',
               marker_color=[colors['primary'] if v == comparison_df['Уверенность'].max() else colors['default'] 
                            for v in comparison_df['Уверенность']]),
        row=1, col=1
    )
    
    # Время
    fig.add_trace(
        go.Bar(name='Время', x=models, y=comparison_df['Время (мс)'],
               text=comparison_df['Время (мс)'].round(1), textposition='auto',
               textfont_color='#ffffff',
               marker_color=[colors['success'] if v == comparison_df['Время (мс)'].min() else colors['default'] 
                            for v in comparison_df['Время (мс)']]),
        row=1, col=2
    )
    
    # Площадь маски
    fig.add_trace(
        go.Bar(name='Площадь', x=models, y=comparison_df['Площадь маски (%)'],
               text=comparison_df['Площадь маски (%)'].round(2), textposition='auto',
               textfont_color='#ffffff',
               marker_color=[colors['secondary'] if v == comparison_df['Площадь маски (%)'].max() else colors['default'] 
                            for v in comparison_df['Площадь маски (%)']]),
        row=2, col=1
    )
    
    # OCR уверенность
    fig.add_trace(
        go.Bar(name='OCR', x=models, y=comparison_df['OCR уверенность'],
               text=comparison_df['OCR уверенность'].round(3), textposition='auto',
               textfont_color='#ffffff',
               marker_color=[colors['warning'] if v == comparison_df['OCR уверенность'].max() else colors['default'] 
                            for v in comparison_df['OCR уверенность']]),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font_color='#8892b0',
        title_font_color='#e0e0e0'
    )
    fig.update_xaxes(tickangle=45, gridcolor='#1a1f2e')
    fig.update_yaxes(gridcolor='#1a1f2e')
    
    return fig

# ==================== ОСНОВНОЙ КОД ====================

# Инициализация
models = available_models()

# Заголовок
st.markdown('<div class="main-header">🚗 ALPR: Сравнение 5 архитектур</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Вариант 12 — Детекция, распознавание, сравнение и анализ ошибок</div>', unsafe_allow_html=True)

# Создание тестовых моделей если нет
if not models:
    st.warning('⚠️ Модели не найдены. Создаю тестовые модели...')
    from src.models import create_model
    for name in MODEL_NAMES:
        model_dir = ROOT / 'checkpoints' / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model = create_model(name, in_channels=1)
        checkpoint = {
            'model_name': name,
            'input_size': 64,
            'model_state': model.state_dict(),
            'best_dice': 0.5,
            'epoch': 1,
            'best_epoch': 1,
            'epochs_requested': 100,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        torch.save(checkpoint, model_dir / 'best.pt')
    models = MODEL_NAMES
    st.success('✅ Тестовые модели созданы!')

ocr_ready, ocr_details = ocr_runtime_status()

# Боковая панель
with st.sidebar:
    st.markdown("### ⚙️ Параметры")
    
    preferred = default_model(models)
    selected_model = st.selectbox(
        '🎯 Модель для отображения',
        models,
        index=models.index(preferred) if preferred in models else 0,
        format_func=lambda name: model_display_names().get(name, name)
    )
    
    st.markdown("---")
    
    compare_mode = st.checkbox('📊 Сравнить все 5 моделей', value=True)
    threshold = st.slider('🎚️ Порог маски', 0.25, 0.80, 0.50, 0.05)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    st.markdown("---")
    st.caption(f'🖥️ Устройство: {device}')
    
    if ocr_ready:
        st.success('✅ OCR готов')
    else:
        st.error('❌ OCR не установлен')
        st.info('Запустите 01_INSTALL.cmd')
    
    st.markdown("---")
    st.markdown("### 📋 Доступные модели")
    for name in models:
        display_name = model_display_names().get(name, name)
        metrics = get_model_metrics(name)
        if metrics:
            st.caption(f'• {display_name}: Dice {metrics.get("best_val_dice", 0):.3f}')
        else:
            st.caption(f'• {display_name}')

# Загрузка изображения
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader('📤 Загрузите фотографию автомобиля', type=['jpg', 'jpeg', 'png'])

with col2:
    demo_dir = ROOT / 'data/demo_samples/images'
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    if not list(demo_dir.glob('*.jpg')):
        try:
            from create_test_data import create_test_dataset
            create_test_dataset()
        except:
            pass
    
    demo_paths = sorted(demo_dir.glob('*.jpg'))
    demo_options = ['- Выберите тестовый пример -'] + [f"{i+1}. {path.stem}" for i, path in enumerate(demo_paths)]
    demo_selection = st.selectbox('🖼️ Тестовый пример', demo_options, index=0)

image = None
source_name = ''
source_bytes = b''

if uploaded is not None:
    source_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(source_bytes)).convert('RGB')
    source_name = uploaded.name
elif demo_selection != '- Выберите тестовый пример -':
    idx = int(demo_selection.split('.')[0]) - 1
    if idx < len(demo_paths):
        path = demo_paths[idx]
        source_bytes = path.read_bytes()
        image = Image.open(path).convert('RGB')
        source_name = path.name

if image is None:
    st.info('📸 Загрузите изображение или выберите тестовый пример')
    st.stop()

# ===================== ОБРАБОТКА =====================

st.markdown("---")

# Основной результат
st.markdown(f"### 🔍 Результат: {model_display_names().get(selected_model, selected_model)}")

checkpoint_path = ROOT / 'checkpoints' / selected_model / 'best.pt'
if not checkpoint_path.exists():
    checkpoint_path = ROOT / 'checkpoints' / selected_model / 'best_weights.pt'

if not checkpoint_path.exists():
    st.error(f'❌ Файл модели не найден: {checkpoint_path}')
else:
    started = time.perf_counter()
    try:
        result = run_alpr(
            np.array(image),
            model_name=selected_model,
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            device=device,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        
        # Отображение результата (без маски)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='📷 Исходное изображение', width='stretch')
            st.caption(f"Размер: {image.size[0]}×{image.size[1]} пикселей")
        
        with col2:
            # Рисуем bbox на изображении (без маски)
            img_with_bbox = draw_bbox_on_image(image, result.get('bbox', []), 
                                              result.get('text', ''), 
                                              result.get('confidence', 0))
            st.image(img_with_bbox, caption='🎯 Результат распознавания', width='stretch')
            
            # Текст номера
            if result.get('text'):
                st.markdown(f'<div class="plate-text">🔤 {result["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ Символы не распознаны</div>', unsafe_allow_html=True)
        
        # Метрики
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('🎯 Уверенность', f"{result.get('confidence', 0):.3f}")
        m2.metric('⏱️ Время', f"{elapsed_ms:.1f} мс")
        m3.metric('📐 Площадь маски', f"{result.get('area_percent', 0):.2f}%")
        m4.metric('📝 OCR', f"{result.get('ocr_confidence', 0):.3f}")
        
        # Детальная информация
        st.markdown("---")
        col1, col2 = st.columns([1, 1.4])
        
        with col1:
            st.markdown("### 📷 Область номера")
            if result.get('crop') is not None and result.get('crop').size > 0:
                st.image(result['crop'], caption='Вырезанная область', width='stretch')
                h, w = result['crop'].shape[:2] if len(result['crop'].shape) >= 2 else (0, 0)
                st.caption(f"Размер: {w}×{h} пикселей")
            else:
                st.warning('⚠️ Область номера не найдена')
        
        with col2:
            st.markdown("### 📊 Детали")
            bbox = result.get('bbox', [])
            if bbox and len(bbox) == 4 and bbox != [0, 0, 0, 0]:
                st.caption(f"📍 **BBox:** ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})")
                st.caption(f"📏 Размер: {bbox[2]-bbox[0]}×{bbox[3]-bbox[1]} пикселей")
            else:
                st.caption("📍 **BBox:** Не найден")
            
            st.caption(f"🔧 **Метод:** {result.get('detection_method', 'segmentation')}")
            st.caption(f"📊 **OCR:** {result.get('ocr_engine', 'unknown')}")
            st.caption(f"📦 **Кандидатов:** {result.get('candidate_count', 1)}")
        
        # Анализ ошибок
        st.markdown("---")
        st.markdown("### 🔍 Анализ ошибок")
        
        errors_found = []
        warnings_found = []
        
        conf = result.get('confidence', 0)
        if conf < 0.3:
            errors_found.append(f'⚠️ Низкая уверенность ({conf:.3f})')
        elif conf < 0.5:
            warnings_found.append(f'⚠️ Средняя уверенность ({conf:.3f})')
        
        if not result.get('text'):
            errors_found.append('⚠️ Текст не распознан')
        
        ocr_status = result.get('ocr_engine', '')
        if 'error' in ocr_status or 'not_available' in ocr_status:
            errors_found.append(f'❌ Ошибка OCR: {ocr_status}')
        
        if result.get('crop') is None or result.get('crop').size == 0:
            errors_found.append('⚠️ Область номера не найдена')
        
        bbox = result.get('bbox', [])
        if bbox == [0, 0, 0, 0] or not bbox:
            errors_found.append('⚠️ BBox не найден')
        
        area = result.get('area_percent', 0)
        if area < 1:
            warnings_found.append(f'⚠️ Малая площадь маски ({area:.2f}%)')
        
        if errors_found:
            st.markdown('<div class="error-box">❌ <b>Найдены ошибки:</b></div>', unsafe_allow_html=True)
            for error in errors_found:
                st.warning(error)
        
        if warnings_found:
            st.markdown('<div class="info-box">⚠️ <b>Предупреждения:</b></div>', unsafe_allow_html=True)
            for warning in warnings_found:
                st.info(warning)
        
        if not errors_found and not warnings_found:
            st.markdown('<div class="success-box">✅ <b>Все проверки пройдены успешно!</b></div>', unsafe_allow_html=True)
        
        if errors_found or warnings_found:
            st.markdown(f"""
            <div class="info-box">
                💡 <b>Рекомендации:</b><br>
                • Попробуйте изменить порог маски (сейчас {threshold:.2f})<br>
                • Используйте другую модель<br>
                • Проверьте качество изображения
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ <b>Ошибка:</b> {str(e)}</div>', unsafe_allow_html=True)

# ===================== СРАВНЕНИЕ МОДЕЛЕЙ =====================

if compare_mode:
    st.markdown("---")
    st.markdown("### 📊 Сравнение всех 5 архитектур")
    st.caption("Каждая модель обрабатывает изображение с одинаковыми параметрами")
    
    with st.spinner('🔄 Обработка всеми моделями...'):
        all_results, all_times, all_errors = process_with_all_models(image, models, threshold, device)
    
    comparison_df = create_comparison_metrics(all_results, all_times, all_errors)
    
    if not comparison_df.empty:
        # Таблица сравнения
        st.dataframe(
            comparison_df[['Модель', 'Уверенность', 'Время (мс)', 'Текст', 'Площадь маски (%)', 'Статус']],
            width='stretch',
            hide_index=True,
            column_config={
                'Модель': st.column_config.TextColumn('Модель', width='medium'),
                'Уверенность': st.column_config.NumberColumn('Уверенность', format='%.3f'),
                'Время (мс)': st.column_config.NumberColumn('Время', format='%.1f'),
                'Текст': st.column_config.TextColumn('Распознанный текст', width='large'),
                'Площадь маски (%)': st.column_config.NumberColumn('Площадь маски', format='%.2f%%'),
                'Статус': st.column_config.TextColumn('Статус')
            }
        )
        
        # Графики сравнения
        st.plotly_chart(create_comparison_chart(comparison_df), use_container_width=True)
        
        # Лучшая модель
        best_models = comparison_df[comparison_df['Статус'] != '❌ Ошибка']
        if not best_models.empty:
            best_conf = best_models.loc[best_models['Уверенность'].idxmax()]
            fastest = best_models.loc[best_models['Время (мс)'].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="best-model-box">
                    <b>🏆 Лучшая по уверенности:</b> {best_conf['Модель']}<br>
                    Уверенность: {best_conf['Уверенность']:.3f}<br>
                    Текст: {best_conf['Текст']}<br>
                    Время: {best_conf['Время (мс)']:.1f} мс
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <b>⚡ Самая быстрая:</b> {fastest['Модель']}<br>
                    Время: {fastest['Время (мс)']:.1f} мс<br>
                    Уверенность: {fastest['Уверенность']:.3f}<br>
                    Текст: {fastest['Текст']}
                </div>
                """, unsafe_allow_html=True)
        
        # Визуальные результаты всех моделей (без маски)
        st.markdown("---")
        st.markdown("### 🖼️ Результаты всех моделей")
        
        num_cols = min(len([m for m in all_results.items() if 'error' not in m[1]]), 5)
        cols = st.columns(num_cols)
        
        for idx, (model_name, result) in enumerate(all_results.items()):
            if idx >= num_cols:
                break
            
            with cols[idx]:
                display_name = model_display_names().get(model_name, model_name)
                st.caption(f'**{display_name}**')
                
                if 'error' in result:
                    st.error(f'❌ Ошибка')
                else:
                    # Показываем результат (без маски)
                    img_array = np.array(image)
                    vis_img = draw_bbox_on_image(
                        img_array,
                        result.get('bbox', []),
                        result.get('text', ''),
                        result.get('confidence', 0)
                    )
                    st.image(vis_img, width='stretch')
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric('Уверенность', f"{result.get('confidence', 0):.3f}")
                    with col_b:
                        st.metric('Время', f"{all_times.get(model_name, 0):.1f} мс")
                    
                    if result.get('text'):
                        st.success(f"📝 {result['text']}")
                    else:
                        st.warning('❌ Не распознан')
        
        # Сохранение результатов
        comparison_log = {
            'timestamp': datetime.now().isoformat(),
            'image': source_name,
            'threshold': threshold,
            'models': {}
        }
        
        for model_name, result in all_results.items():
            if 'error' not in result:
                comparison_log['models'][model_name] = {
                    'confidence': result.get('confidence', 0),
                    'time_ms': all_times.get(model_name, 0),
                    'text': result.get('text', ''),
                    'ocr_confidence': result.get('ocr_confidence', 0),
                    'area_percent': result.get('area_percent', 0),
                    'bbox': result.get('bbox', [])
                }
        
        append_comparison_log(comparison_log)

# ===================== СТАТИСТИКА =====================

st.markdown("---")
st.markdown("### 📈 Статистика работы моделей")

comparison_history = read_comparison_history()
if not comparison_history.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('📊 Всего сравнений', len(comparison_history))
    col2.metric('📸 Уникальных изображений', comparison_history['image'].nunique())
    
    avg_conf = comparison_history['confidence'].mean() if not comparison_history.empty else 0
    col3.metric('🎯 Средняя уверенность', f"{avg_conf:.3f}")
    
    avg_time = comparison_history['time_ms'].mean() if not comparison_history.empty else 0
    col4.metric('⏱️ Среднее время', f"{avg_time:.1f} мс")
    
    model_stats = create_model_statistics()
    if model_stats is not None and not model_stats.empty:
        st.dataframe(
            model_stats,
            width='stretch',
            hide_index=True,
            column_config={
                'Модель': st.column_config.TextColumn('Модель'),
                'Средняя уверенность': st.column_config.NumberColumn('Средняя уверенность', format='%.3f'),
                'Макс уверенность': st.column_config.NumberColumn('Макс уверенность', format='%.3f'),
                'Среднее время (мс)': st.column_config.NumberColumn('Среднее время', format='%.1f'),
                'Кол-во обработанных': st.column_config.NumberColumn('Обработано', format='%d'),
                'Успешных распознаваний': st.column_config.NumberColumn('Успешно', format='%d'),
                'Точность': st.column_config.NumberColumn('Точность', format='%.1f%%')
            }
        )
        
        if not model_stats.empty:
            best_stats = model_stats.loc[model_stats['Средняя уверенность'].idxmax()]
            st.markdown(f"""
            <div class="best-model-box">
                <b>🏆 Лучшая по средней уверенности:</b> {best_stats['Модель']}<br>
                Средняя уверенность: {best_stats['Средняя уверенность']:.3f}<br>
                Точность: {best_stats['Точность']:.1%}<br>
                Обработано: {int(best_stats['Кол-во обработанных'])} изображений
            </div>
            """, unsafe_allow_html=True)

# ===================== ФУТЕР =====================

st.markdown("---")
st.caption('🚗 **ALPR System v1.0** — Сравнение 5 архитектур нейронных сетей')
st.caption(f'📁 Логи: {LOG_PATH}')
st.caption('📋 **Задание:** Детекция, распознавание, сравнение архитектур, анализ ошибок')