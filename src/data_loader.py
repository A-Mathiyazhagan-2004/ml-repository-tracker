import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def load_data(file_path, target_column, test_size=0.2, random_state=42):

    if file_path is None:
        df = load_breast_cancer(as_frame=True)
        x = df.data
        y = df.target
        
    else:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        df = pd.read_csv(path)

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in the dataset."
                f" Available columns: {list(df.columns)}"
            )

        if df.empty:
            raise ValueError("The dataset is empty.")

        if df[target_column].isna().any():
            raise ValueError(f"Target column '{target_column}' contains missing values.")
    
        x = df.drop(columns=[target_column])
        y = df[target_column]

        if x.shape[1] == 0:
            raise ValueError("The dataset must contain at least one feature column.")

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
            raise ValueError("All feature columns must be numeric.")

        if x.isna().any().any():
            raise ValueError("Feature columns contain missing values.")

        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y), index=df.index)

        print(f"Target encoding: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    if y.nunique() != 2:
        raise ValueError("The target values must be binary to be supported.")
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test