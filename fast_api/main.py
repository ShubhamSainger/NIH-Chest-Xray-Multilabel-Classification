import sys
sys.path.append("/app")

# Global modules
from fastapi import FastAPI, UploadFile
from pickle import load

# Local Modules
from data_loading_inference import load_data
from model import model
from configs import configs

# API

app = FastAPI()

model = model.network()
model.load_weights(configs.MODEL_WEIGHTS_PATH)


with open(configs.DISEASE_ENCODINGS_PATH, mode="rb") as f:
    diseases = load(f)


# Methods

@app.get("/")
def root():
    return {"Response" : "This API is working on root"}


@app.post("/Dignosis")
def upload(file: UploadFile):

    image = load_data.load(file)

    if image is None:
        return {"file_type_error": "Please upload a image instead of {}".format([file.content_type])}
    
    disease_prob = model.predict(image)[0].tolist()
    disease_prob = [round(x, 3) for x in disease_prob]

    return dict(zip(diseases,disease_prob))
