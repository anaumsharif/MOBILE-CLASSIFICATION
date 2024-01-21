from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf




app = FastAPI()

# creating global variable of the model
MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ['avg', 'good', 'poor']


# @app.get("/ping")
# async def ping():
#     return "Hello ,I am alive"
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    # since our model needs batch of images np.expand_dims helps in expanding or increasing the dimension of the images
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    # index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class,confidence)
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

    pass

if __name__ == "__main__" :
    uvicorn.run(app,host='localhost',port=8000)

