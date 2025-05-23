import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fuzzywuzzy import process
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DebertaV2Tokenizer,
    AutoTokenizer,
)
import torch
from torch.utils.data import Dataset
import logging
import os

# Set tokenizers parallelism to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class RegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
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

def train_model(
    train_path: str,
    model_name: str = "microsoft/deberta-v3-base",
    checkpoint_path: str = None,
    val_size: float = 0.1,
    output_dir: str = "./model",
    # Основные параметры обучения
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 64,
    # Параметры оптимизации
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    # Настройки оценки и сохранения
    eval_strategy: str = "epoch",
    eval_steps: int = None,
    save_strategy: str = "epoch",
    save_steps: int = 500,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "mse",
    greater_is_better: bool = False,
    # Настройки логирования
    logging_dir: str = "./logs",
    logging_steps: int = 30,
    logging_first_step: bool = True,
    report_to: str = "none",
    # Аппаратные настройки
    fp16: bool = None,
    no_cuda: bool = False,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 4,
    # Другие настройки
    disable_tqdm: bool = False,
    remove_unused_columns: bool = True,
    label_names: list = ["labels"],
    group_by_length: bool = False,
    **kwargs
):
    # Автоматическая настройка fp16
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    # Инициализация TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=report_to,
        disable_tqdm=disable_tqdm,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        warmup_steps=warmup_steps,
        logging_first_step=logging_first_step,
        no_cuda=no_cuda,
        remove_unused_columns=remove_unused_columns,
        label_names=label_names,
        group_by_length=group_by_length,
        **kwargs
    )
    
    # Загрузка и разделение данных
    logger.info("Loading and preparing data...")
    df = pd.read_csv(train_path)
    
    # Предобработка данных
    df = preprocess_data(df)
    
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=42
    )
    
    # Нормализация log_salary_from для улучшения обучения
    logger.info("Normalizing salary values...")
    scaler = StandardScaler()
    train_df["normalized_log_salary"] = scaler.fit_transform(train_df[["log_salary_from"]])
    val_df["normalized_log_salary"] = scaler.transform(val_df[["log_salary_from"]])
    
    # Сохранение параметров нормализации для последующего использования
    normalization_params = {
        "mean": float(scaler.mean_[0]),
        "scale": float(scaler.scale_[0])
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "normalization_params.json"), "w") as f:
        json.dump(normalization_params, f)
    logger.info(f"Normalization parameters saved: mean={normalization_params['mean']:.4f}, scale={normalization_params['scale']:.4f}")
    
    # Проверка наличия чекпоинта модели
    if checkpoint_path:
        checkpoint_path = os.path.join(output_dir, checkpoint_path)
        if os.path.exists(checkpoint_path):
            logger.info(f"Найден чекпоинт модели в {output_dir}. Продолжаем обучение...")
            model = AutoModelForSequenceClassification.from_pretrained(
                output_dir,
                num_labels=1,
                problem_type="regression"
            )
            logger.info("Модель и токенизатор успешно загружены из чекпоинта.")
    else:
        # Инициализация новой модели
        logger.info("Чекпоинт не найден. Инициализация новой модели...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
            ignore_mismatched_sizes=True  # Add this parameter to handle classifier size mismatch
        )
        logger.info("Модель успешно инициализирована с новым классификатором для регрессии.")

    if model_name == "microsoft/deberta-v3-base":
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Подготовка датасетов с обогащенными данными
    train_texts = train_df.apply(
        lambda row: f"{row['title']} "
                    f"{'неизвестно' if pd.isna(row['skills']) else row['skills']} "
                    f"{'неизвестно' if not row['location'] else row['location']} "
                    f"{'неизвестно' if not row['region'] else row['region']} "
                    f"{'неизвестно' if int(row['population'])==0 else int(row['population'])} "
                    f"{'неизвестно' if int(row['mean_salary'])==0 else int(row['mean_salary'])} "
                    f"{int(row['experience_from'])} "
                    f"{row['description'][:512]}",
        axis=1
    )
    
    val_texts = val_df.apply(
        lambda row: f"{row['title']} "
                    f"{'неизвестно' if pd.isna(row['skills']) else row['skills']} "
                    f"{'неизвестно' if not row['location'] else row['location']} "
                    f"{'неизвестно' if not row['region'] else row['region']} "
                    f"{'неизвестно' if int(row['population'])==0 else int(row['population'])} "
                    f"{'неизвестно' if int(row['mean_salary'])==0 else int(row['mean_salary'])} "
                    f"{int(row['experience_from'])} "
                    f"{row['description'][:512]}",
        axis=1
    )

    print("Examples Valid Samples:")
    for i, sample in enumerate(val_texts.tolist()[:3], 1):
        print(f"\nExample {i}:\n{sample[:512]}\n{'-' * 80}")

    # print(train_df["log_salary_from"])
    # print(val_df["log_salary_from"])
    
    train_dataset = RegressionDataset(
        train_texts,
        train_df["normalized_log_salary"],  # Используем нормализованные значения
        tokenizer,
        max_length=512
    )
    
    val_dataset = RegressionDataset(
        val_texts,
        val_df["normalized_log_salary"],  # Используем нормализованные значения
        tokenizer,
        max_length=512
    )
    
    # Метрики для регрессии (адаптированы для работы с нормализованными значениями)
    def compute_metrics(p):
        preds = p.predictions.squeeze()
        labels = p.label_ids
        
        # Денормализация для расчета метрик в исходном масштабе
        preds_denorm = preds * normalization_params["scale"] + normalization_params["mean"]
        labels_denorm = labels * normalization_params["scale"] + normalization_params["mean"]
        
        return {
            "mse": mean_squared_error(labels, preds),  # Нормализованная MSE
            "mae": mean_absolute_error(labels, preds),  # Нормализованная MAE
            "r2": r2_score(labels, preds),  # R2 одинаков для нормализованных и исходных данных
            "raw_mse": mean_squared_error(labels_denorm, preds_denorm),  # MSE в исходном масштабе
            "raw_mae": mean_absolute_error(labels_denorm, preds_denorm)  # MAE в исходном масштабе
        }
    
    # Определяем кастомную функцию потерь для оптимизации R2
    def r2_loss_fn(outputs, labels, num_items_in_batch):
        # Извлекаем предсказания из outputs
        if isinstance(outputs, dict):
            logits = outputs["logits"].squeeze()
        else:
            logits = outputs[0].squeeze()
        
        # Получаем метки
        labels = labels.squeeze()
        
        # Расчет среднего значения меток (target)
        mean_labels = torch.mean(labels)
        
        # Расчет общей суммы квадратов (total sum of squares)
        total_sum_squares = torch.sum((labels - mean_labels) ** 2)
        
        # Расчет суммы квадратов остатков (residual sum of squares)
        residual_sum_squares = torch.sum((labels - logits) ** 2)
        
        # Расчет R2: R² = 1 - RSS/TSS
        r2 = 1 - (residual_sum_squares / (total_sum_squares + 1e-8))  # добавляем эпсилон для численной стабильности
        
        # Преобразуем в функцию потерь (чтобы минимизировать)
        loss = 1 - r2  # минимизация этой величины эквивалентна максимизации R2
        
        # Проверка на NaN и Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"R2 loss is {loss}, using MSE instead")
            loss = torch.mean((labels - logits) ** 2)  # Используем MSE в случае проблем
            
        return loss
    
    # Обучение
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        compute_loss_func=r2_loss_fn,  # Используем кастомную функцию потерь
    )
    
    trainer.train()
    
    # Сохранение модели
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return trainer


# Пример использования
if __name__ == "__main__":
    trainer = train_model(
        train_path="data/train.csv",
        model_name="microsoft/deberta-v3-base", # zloelias/rubert-tiny2-kinopoisk-reviews-finetuned-clf microsoft/deberta-v3-base # cointegrated/rubert-tiny-sentiment-balanced
        checkpoint_path=None,
        val_size=0.01,
        output_dir="./regression_model",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01
    )
