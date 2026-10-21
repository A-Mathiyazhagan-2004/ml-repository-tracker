from models import models
from data_loader import load_data
import time

def train_models(x_train, y_train):
    trained_models = {}
    training_times = {}
    for model_name, model in models().items():
        print(f"Training {model_name}")
        start_time = time.perf_counter()
        trained_models[model_name] = model.fit(x_train, y_train)
        end_time = time.perf_counter()
        training_times[model_name] = end_time - start_time
        print(f"Training time: {training_times[model_name]:.4f} seconds")
    return trained_models, training_times