import numpy as np
from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

@app.get("/ping")
async def ping():
    return ("Hello Server is Alive")

Model = tf.keras.models.load_model("E:/Python_Data_Science/Tomato_Disease_DL/Project/Training/models")
Class_names = ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
def read_file_as_image(data):#--> (np.ndarray
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
file: UploadFile = File(...)
):
    images= await read_file_as_image(await file.read)
    image_batch = np.expand_dims(images, 0)
    predictions = Model.predict(image_batch)
    index = np.argmax(predictions[0])
    predicted_class = Class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app, host='localhost',port=8000)
