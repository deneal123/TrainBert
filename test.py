import pandas as pd
from sklearn.model_selection import train_test_split
val_size = 0.1
train_path = "data/train.csv"

df = pd.read_csv(train_path)
train_df, val_df = train_test_split(
    df,
    test_size=val_size,
    stratify=df['rate'],
    random_state=42
)

print(train_df["rate"] - 1)
