import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import numpy as np

def train_model(minnie_array, miyeon_array, shuhua_array, yuqi_array, soyeon_array):
    print(f'Minnie: {len(minnie_array)}')
    print(f'Miyeon: {len(miyeon_array)}')
    print(f'Shuhua: {len(shuhua_array)}')
    print(f'Yuqi: {len(yuqi_array)}')
    print(f'Soyeon: {len(soyeon_array)}')

    x = np.concatenate([minnie_array, miyeon_array, shuhua_array, yuqi_array, soyeon_array], axis=0)
    y = np.array([0] * len(minnie_array) + [1] * len(miyeon_array) + [2] * len(shuhua_array) + [3] * len(yuqi_array) + [4] * len(soyeon_array))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=16)

    model = SVC(C=10, kernel='rbf', gamma="scale", probability=True)
    model.fit(x_train, y_train)
    
    joblib.dump(model, 'Model/model_svc.pkl')
    y_pred = model.predict(x_test)

    return model, y_test, y_pred