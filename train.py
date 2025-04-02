import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset
import logging

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# 1. Класс датасета
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    

# 2. Функция обучения
def train_model(
    train_path: str,
    model_name: str = "microsoft/deberta-v2-xxlarge",
    val_size: float = 0.1,
    output_dir: str = "./model",
    # Основные параметры обучения
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
    # Параметры оптимизации
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    # Настройки оценки и сохранения
    evaluation_strategy: str = "epoch",
    eval_steps: int = None,
    save_strategy: str = "epoch",
    save_steps: int = 500,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    # Настройки логирования
    logging_dir: str = "./logs",
    logging_steps: int = 100,
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
    """
    Обучение модели классификации отзывов
    
    Параметры:
    train_path (str): Путь к тренировочным данным
    model_name (str): Название предобученной модели
    val_size (float): Доля валидационной выборки (0-1)
    output_dir (str): Директория для сохранения модели
    num_train_epochs (int): Количество эпох обучения
    learning_rate (float): Скорость обучения
    per_device_train_batch_size (int): Размер батча для обучения
    per_device_eval_batch_size (int): Размер батча для оценки
    weight_decay (float): Коэффициент L2-регуляризации
    gradient_accumulation_steps (int): Шаги накопления градиентов
    warmup_steps (int): Шаги прогрева для обучения
    evaluation_strategy (str): Стратегия оценки (no, steps, epoch)
    eval_steps (int): Шаги между оценкой (если strategy=steps)
    save_strategy (str): Стратегия сохранения (no, steps, epoch)
    save_steps (int): Шаги между сохранениями (если strategy=steps)
    load_best_model_at_end (bool): Загружать лучшую модель в конце
    metric_for_best_model (str): Метрика для выбора лучшей модели
    greater_is_better (bool): Ориентация метрики (больше=лучше)
    logging_dir (str): Директория для логов
    logging_steps (int): Частота логирования (в шагах)
    logging_first_step (bool): Логировать первый шаг
    report_to (str): Куда отправлять логи (tensorboard, wandb)
    fp16 (bool): Использовать смешанную точность
    no_cuda (bool): Отключить CUDA
    gradient_checkpointing (bool): Экономия памяти
    dataloader_num_workers (int): Количество воркеров загрузки данных
    disable_tqdm (bool): Отключить индикатор прогресса
    remove_unused_columns (bool): Удалять неиспользуемые колонки
    label_names (list): Названия полей с метками
    group_by_length (bool): Группировать по длине последовательностей
    kwargs: Дополнительные аргументы для TrainingArguments
    """
    
    # Автоматическая настройка fp16
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    # Инициализация TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
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
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df['rate'],
        random_state=42
    )
    
    # Инициализация модели
    logger.info("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,
        id2label={i: str(i+1) for i in range(5)},
        label2id={str(i+1): i for i in range(5)}
    )
    
    # Подготовка датасетов
    train_dataset = ReviewDataset(
        train_df["text"],
        train_df["rate"] - 1,  # Конвертация 1-5 в 0-4
        tokenizer
    )
    
    val_dataset = ReviewDataset(
        val_df["text"],
        val_df["rate"] - 1,
        tokenizer
    )
    
    # Метрики
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": (preds == p.label_ids).mean(),
            "f1": f1_score(p.label_ids, preds, average='macro')
        }
    
    # Обучение
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Сохранение модели
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return trainer


# 4. Пример использования
if __name__ == "__main__":
    # Обучение модели
    trainer = train_model(
        train_path="data/train.csv",
        model_name="microsoft/deberta-v2-xxlarge",
        val_size=0.1,
        output_dir="./model",
        num_epochs=3,
        batch_size=4
    )
