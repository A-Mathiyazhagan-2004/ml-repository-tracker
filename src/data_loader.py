import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, target_column, test_size=0.2, random_state=42):
    
    df = pd.read_csv(file_path)
    
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the dataset."
            f" Available columns: {list(df.columns)}")
    if df.empty:
        raise ValueError("The dataset is empty.")
    if df[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")
    
    x = df.drop(columns=[target_column])
    y = df[target_column]

    if y.nunique() != 2:
        raise ValueError("The target values must be binary to be supported.")
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test