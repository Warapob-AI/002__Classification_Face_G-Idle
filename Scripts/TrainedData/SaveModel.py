import joblib

def save_model(model, filename): 
    joblib.dump(f'Model/{model}', f'{filename}')

