from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def models():

    model_configs = {}

    for c in [0.1, 1, 10]:
        model_configs[f"Logistic Regression (C={c})"] = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=42))

    for h in [(50,), (100,), (50, 50)]:
        model_configs[f"MLP Classifier (hidden_layer_sizes={h})"] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, max_iter=1000, random_state=42))

    for n in [50, 100, 200]:
        model_configs[f"Random Forest (n_estimators={n})"] = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n, random_state=42))

    return model_configs