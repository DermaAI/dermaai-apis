import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow-cpu as tf
from PIL import Image
from io import BytesIO
# load api
app = FastAPI()

# load model for prediction
# Load the TFLite model and allocate tensors.
interpreter1 = tf.lite.Interpreter(model_path="models/main-model1.tflite")
interpreter1.allocate_tensors()

# Get input and output tensors.
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

# load model for prediction
# Load the TFLite model and allocate tensors.
interpreter2 = tf.lite.Interpreter(model_path="models/second-model.tflite")
interpreter2.allocate_tensors()

# Get input and output tensors.
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

# method to predict main disease
def mainPredict(image: Image.Image):
    image = np.asarray(image.resize((224, 224)), dtype=np.float32)[..., :3]
    image = np.expand_dims(image, 0)
    images = np.vstack([image])

    # # Test the model on random input data.
    input_shape = input_details1[0]['shape']
    interpreter1.set_tensor(input_details1[0]['index'], images)
    interpreter1.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter1.get_tensor(output_details1[0]['index'])    
    return {
        "prediction": str(np.argmax(output_data)),
        "all": str(output_data)
    }

# method to predict cancer 
def cancerPredict(image: Image.Image):
    image = np.asarray(image.resize((224, 224)), dtype=np.float32)[..., :3]
    image = np.expand_dims(image, 0)
    images = np.vstack([image])

    # # Test the model on random input data.
    input_shape = input_details2[0]['shape']
    interpreter2.set_tensor(input_details2[0]['index'], images)
    interpreter2.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter2.get_tensor(output_details2[0]['index'])    
    return {
        "prediction": str(np.argmax(output_data)),
        "all": str(output_data)
    }



# sample home page
@app.get('/')
def index():
    return {'message': 'Hello, World'}

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# main model prediction
@app.post("/predict/main")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = mainPredict(image)
    return prediction

# cancer model prediction
@app.post("/predict/cancer")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = cancerPredict(image)
    return prediction


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload


