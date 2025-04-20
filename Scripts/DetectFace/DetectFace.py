import numpy as np
import cv2 

prototxt_path = "Pre-Trained/deploy.prototxt"
model_path = "Pre-Trained/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_face(image):
    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            padding = 20
            x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding)

            face = image[y1:y2, x1:x2]
            return face 

    return None 
