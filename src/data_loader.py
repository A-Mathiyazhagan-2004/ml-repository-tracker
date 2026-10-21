from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

def load_data(test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)