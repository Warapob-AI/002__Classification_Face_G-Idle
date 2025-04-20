import joblib

def load_model(model):
    model_array = joblib.load(f"{model}")
    return model_array