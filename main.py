from Scripts.LoadFolder.LoadImage import load_folder_image
from Scripts.TrainedData.SVM import train_model
from Scripts.TrainedData.SaveModel import save_model
from Scripts.TrainedData.LoadModel import load_model
from Scripts.EvaluationData.Classification import classification
from Scripts.EvaluationData.Predict import predict_new_image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import shutil
import os

if not os.path.exists('Model/model_svm.pkl'):
    training_minnie = load_folder_image('Dataset/Training/minnie_face')
    training_miyeon = load_folder_image('Dataset/Training/miyeon_face')
    training_shuhua = load_folder_image('Dataset/Training/shuhua_face')
    training_soyeon = load_folder_image('Dataset/Training/soyeon_face')
    training_yuqi = load_folder_image('Dataset/Training/yuqi_face')

    model_svm, y_test, y_pred = train_model(training_minnie, training_miyeon, training_shuhua, training_soyeon, training_yuqi)
    classification(y_test, y_pred)
    save_model(model_svm, 'Model/model_svm.pkl')


model = load_model('Model/model_svm.pkl')

app = FastAPI()

@app.post("/")
async def predict_text(file: UploadFile = File(...)):
    # Save the uploaded file
    image_path = f"{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_new_image(image_path, model)

    return JSONResponse(content={"result": result})