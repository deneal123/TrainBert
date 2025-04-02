import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    pipeline
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
    

# 3. Функция инференса
def inference_model(
    test_path: str,
    model_path: str = "./model",
    output_path: str = "submission.csv",
    use_pipeline: bool = True,
    batch_size: int = 16
):
    """
    Предсказание на тестовых данных
    Параметры:
    test_path - путь к тестовым данным
    model_path - путь к сохраненной модели
    output_path - путь для сохранения результатов
    use_pipeline - использовать pipeline для инференса
    batch_size - размер батча для предсказаний
    """
    
    # Загрузка данных
    logger.info("Loading test data...")
    test_df = pd.read_csv(test_path)
    
    # Загрузка модели
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if use_pipeline:
        # Инференс через pipeline
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=batch_size,
            return_all_scores=False
        )
        
        logger.info("Running pipeline inference...")
        predictions = classifier(
            test_df["text"].tolist(),
            truncation=True,
            max_length=256
        )
        
        # Обработка результатов
        labels = [int(pred['label']) + 1 for pred in predictions]
        
    else:
        # Инференс через Trainer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        trainer = Trainer(model=model)
        
        test_dataset = ReviewDataset(test_df["text"], None, tokenizer)
        predictions = trainer.predict(test_dataset)
        labels = np.argmax(predictions.predictions, axis=1) + 1
    
    # Сохранение результатов
    test_df['rate'] = labels
    test_df[['index', 'rate']].to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


# 4. Пример использования
if __name__ == "__main__":
    
    # Предсказание на тестовых данных
    inference_model(
        test_path="data/test.csv",
        model_path="./model",
        output_path="submission.csv",
        use_pipeline=True,
        batch_size=16
    )
