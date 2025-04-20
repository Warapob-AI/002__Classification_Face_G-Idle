from Scripts.DetectFace.DetectFace import detect_face
from Scripts.DetectFace.ExtractFeatures import extract_features

import numpy as np
import cv2 
import os

def load_folder_image(folder_path):
    features = []
    for data in os.listdir(folder_path):
        file_path = os.path.join(folder_path, data)
        image = cv2.imread(file_path)

        if image is None:
            print(f'❌ NOT FOUND IMAGE: {file_path}')
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        face = detect_face(image)
        if face is None:
            print(f'❌ No Face Found: {file_path}')
            os.remove(file_path)

        face_features = extract_features(face)
        features.append(face_features)

    return np.array(features)
