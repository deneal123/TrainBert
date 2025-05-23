import pandas as pd
import numpy as np
import re
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
from torch.utils.data import Dataset
import logging

# Set tokenizers parallelism to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('inference.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Класс датасета для регрессии
class RegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=2048):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        self.labels = labels.tolist() if labels is not None else None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Preprocessing functions
def preprocess_location(location: str) -> str:
    if not location:
        return 'Другое'

    # Удаление информации о метро
    location = re.sub(r'\(м\..*?\)', '', location, flags=re.IGNORECASE)
    location = re.sub(r'\(метро.*?\)', '', location, flags=re.IGNORECASE)
    location = re.sub(r'м\..*', '', location, flags=re.IGNORECASE)

    # Удаление информации о странах и других нерелевантных данных в скобках
    location = re.sub(r'\(.*?(область|край|республика|район|АО|Югра|Сербия|Армения|Узбекистан|Беларусь).*?\)', '', location, flags=re.IGNORECASE)

    # Удаление лишних пробелов и запятых
    location = re.sub(r'\s*,\s*.*', '', location)  # Удаляем всё после запятой
    location = re.sub(r'\s+', ' ', location).strip()  # Удаляем лишние пробелы

    # Удаляем "Московская область", сохраняя остальной текст
    location = re.sub(r'\s*\(?Московская область\)?\s*', ' ', location, flags=re.IGNORECASE).strip()
    location = re.sub(r'\s*\(?Кировская область\)?\s*', ' ', location, flags=re.IGNORECASE).strip()

    # Стандартизация названий
    replacements = {
        r'Санкт-Петербург.*': 'Санкт-Петербург',
        r'Москва.*': 'Москва',
        r'неизвестно': 'Другое',
        r'.*Республика Саха \(Якутия\).*': 'Якутия',
    }

    for pattern, replacement in replacements.items():
        if re.search(pattern, location, re.IGNORECASE):
            return replacement

    # Возврат результата
    return location if location else 'Другое'


def normalize_region(region):
    if pd.isna(region):  # Проверка на NaN или None
        return None
    
    # Удаляем лишние символы
    region = re.sub(r'\[.*?\]', '', region)  # Удаляем всё в квадратных скобках
    region = re.sub(r'\\xa0', ' ', region)  # Заменяем \xa0 на пробел
    region = re.sub(r'\s+', ' ', region).strip()  # Убираем лишние пробелы
    
    # Замены для устаревших или неправильных названий
    replacements = {
        'Республика Крым/Автономная Республика Крым': 'Республика Крым',
        'город федерального значения Севастополь/Севастопольский городской совет': 'Севастополь',
        'Кемеровская область — Кузбасс': 'Кемеровская область',
        'Республика Саха (Якутия)': 'Якутия',
        'Чувашская Республика': 'Чувашия',
        'Республика Башкортостан': 'Башкортостан',
        'Республика Татарстан': 'Татарстан',
        'Республика Коми': 'Коми',
        'Республика Карелия': 'Карелия',
        'Республика Алтай': 'Алтай',
        'Северная Осетия — Алания': 'Северная Осетия',
        'Чеченская Республика': 'Чечня',
        'Запорожская': 'Запорожская область',
        'Херсонская': 'Херсонская область',
        'Донецкая область': 'Донецкая область',
        'Луганская область': 'Луганская область'
    }
    
    # Применяем замены
    for old, new in replacements.items():
        if old in region:
            region = new
            break
    
    return region


def clean_description(text: str) -> str:
    if pd.isna(text) or text is None:
        return ''
    # Удаление всех специальных символов кроме букв и пробелов
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text)
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    text = text[:512]
    
    return text


def preprocess_data(df, cities_path='./data/cities.csv', region_salary_path='./data/region_salary_mapping.json'):
    """Выполняет полную предобработку данных"""
    logger.info("Starting data preprocessing...")
    
    # Обработка локации
    df['location'] = df['location'].fillna('неизвестно')
    df['location'] = df['location'].apply(preprocess_location)
    
    # Стандартизация городов
    city_replacements = {
        'Зеленоград': 'Москва',
        'Орел': 'Орёл',
        'Королев': 'Королёв',
        'Другое': 'Неизвестно',
        'неизвестно': 'Неизвестно',
        'Вся РФ': 'Неизвестно',
        'удаленно': 'Неизвестно',
        'Россия': 'Неизвестно',
        'Обь': 'Неизвестно',
        'Анна': 'Неизвестно',
        'Ленинский )': 'Ленинский'
    }
    df['location'] = df['location'].replace(city_replacements)
    
    # Добавление данных о городах и регионах
    try:
        cities_df = pd.read_csv(cities_path)
        city_info = cities_df.set_index('location').to_dict(orient='index')
        df['population'] = df['location'].apply(lambda x: city_info.get(x, {}).get('population'))
        df['region'] = df['location'].apply(lambda x: city_info.get(x, {}).get('region'))
        df['region'] = df['region'].apply(normalize_region)
        
        # Добавление данных о средней зарплате в регионе
        try:
            with open(region_salary_path, 'r', encoding='utf-8') as f:
                region_salary_dict = json.load(f)
                df['mean_salary'] = df['region'].map(region_salary_dict)
                df['mean_salary'] = df['mean_salary'].fillna(0)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load region salary data: {e}")
            df['mean_salary'] = 0
    except FileNotFoundError as e:
        logger.warning(f"Could not load cities data: {e}")
        df['population'] = 0
        df['region'] = 'неизвестно'
        df['mean_salary'] = 0
    
    # Обработка пропущенных значений
    df['population'] = df['population'].fillna(0)
    df['population'] = df['population'].apply(lambda x: 0 if x > 20000000 else x)
    
    # Очистка описания
    df['description'] = df['description'].fillna('неизвестно')
    df['description'] = df['description'].apply(clean_description)
    
    logger.info("Data preprocessing completed.")
    return df


# Функция инференса для регрессии
def inference_regression(
    test_path: str,
    model_id: str = "microsoft/deberta-v3-base",
    model_path: str = "./regression_model",
    checkpoint_path: str = None,
    output_path: str = "regression_predictions.csv",
    batch_size: int = 16
):
    """
    Предсказание на тестовых данных для регрессионной модели
    
    Параметры:
    test_path - путь к тестовым данным
    model_path - путь к сохраненной модели
    output_path - путь для сохранения результатов
    batch_size - размер батча для предсказаний
    """
    
    # Загрузка данных
    logger.info("Loading test data...")
    test_df = pd.read_csv(test_path)
    
    # Предобработка данных
    logger.info("Preprocessing test data...")
    test_df = preprocess_data(test_df)
    
    # Загрузка модели и токенизатора
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Загрузка параметров нормализации
    norm_params_path = os.path.join(model_path, "normalization_params.json")
    try:
        with open(norm_params_path, 'r') as f:
            normalization_params = json.load(f)
        logger.info(f"Loaded normalization parameters: mean={normalization_params['mean']:.4f}, scale={normalization_params['scale']:.4f}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load normalization parameters: {e}. Using default values.")
        normalization_params = {"mean": 0.0, "scale": 1.0}
    
    # Подготовка текстовых данных в том же формате, что и при обучении
    test_texts = test_df.apply(
        lambda row: f"Название профессии: {row['title']} "
                    f"Навыки: {'неизвестно' if pd.isna(row['skills']) else row['skills']} "
                    f"Город: {'неизвестно' if not row['location'] else row['location']} "
                    # f"{'неизвестно' if not row['region'] else row['region']} "
                    f"Население города: {'неизвестно' if int(row['population'])==0 else int(row['population'])} "
                    f"Средняя зарплата: {'неизвестно' if int(row['mean_salary'])==0 else int(row['mean_salary'])} "
                    f"Опыт работы: {int(row['experience_from'])} "
                    f"Описание: {row['description'][:512]}",
        axis=1
    )
    
    # Инференс через pipeline
    logger.info("Setting up regression pipeline...")
    regressor = pipeline(
        "text-classification",
        model=os.path.join(model_path, checkpoint_path),
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size
    )
    
    logger.info("Running regression inference...")
    results = []
    
    # Обрабатываем данные порциями, чтобы избежать проблем с памятью
    predictions = regressor(test_texts.tolist(), truncation=True, max_length=512)
        
    # Извлекаем значения из предсказаний
    batch_results = [float(pred["score"]) for pred in predictions]
    results.extend(batch_results)
    
    # Денормализация и конвертация логарифмических предсказаний обратно
    logger.info("Processing predictions...")
    test_df['normalized_prediction'] = results
    
    # Денормализация предсказаний
    test_df['log_prediction'] = test_df['normalized_prediction'] * normalization_params['scale'] + normalization_params['mean']
    
    # ВАЖНО: Не выполняем преобразование из логарифмического масштаба
    # Так как ожидаемые значения должны быть в диапазоне 3-5 (log-scale)
    test_df['prediction'] = test_df['log_prediction']  # Убираем np.exp() здесь
    
    # Сохранение результатов
    logger.info(f"Saving predictions to {output_path}")
    output_df = test_df[['prediction']]
    
    # Сохраняем с индексом, который будет в первой колонке
    output_df.to_csv(output_path, index=True)
    logger.info(f"Predictions saved to {output_path}")
    
    return test_df


# Пример использования
if __name__ == "__main__":
    predictions_df = inference_regression(
        test_path="data/test.csv",
        model_id="microsoft/deberta-v3-base",
        model_path="./regression_model",
        checkpoint_path="./checkpoint-5160",
        output_path="salary_predictions.csv",
        batch_size=16
    )
    
    # Вывод статистики по предсказаниям
    if 'prediction' in predictions_df.columns:
        logger.info("Prediction statistics:")
        logger.info(f"Mean: {predictions_df['prediction'].mean()}")
        logger.info(f"Median: {predictions_df['prediction'].median()}")
        logger.info(f"Min: {predictions_df['prediction'].min()}")
        logger.info(f"Max: {predictions_df['prediction'].max()}")
