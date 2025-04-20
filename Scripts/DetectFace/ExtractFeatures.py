from deepface import DeepFace

import numpy as np
import cv2

def extract_features(image):
    image = cv2.resize(image, (256, 256)) 
    
    try:
        features = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
        print("📌 Features Output:", features)
        
        if isinstance(features, list) and len(features) > 0:
            return np.array(features[0]['embedding'])
        else:
            print("⚠️ No embedding found")
            return None
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return None