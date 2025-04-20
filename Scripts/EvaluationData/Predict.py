from Scripts.DetectFace.DetectFace import detect_face
from Scripts.DetectFace.ExtractFeatures import extract_features

import numpy as np
import cv2

def predict_new_image(image_path, model):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ ไม่พบไฟล์: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    face = detect_face(image)
    if face is None:
        print("❌ ไม่พบใบหน้าในภาพ")
        return
    
    face_features = extract_features(face)
    face_features = face_features.reshape(1, -1) 
    
    predicted_label = model.predict(face_features)[0] 

    labels = ['Minnie', 'Miyeon', 'Shuhua', 'Yuqi', 'Soyeon']
    label = labels[predicted_label] if predicted_label < len(labels) else "Unknown"

    print(f"✅ โมเดลทำนายว่าเป็น: {label}")
    return f'เธอคือ "{label} ✅"'