from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pickle
from io import BytesIO
import base64, json
from PIL import Image
import os

app = FastAPI()
client = MongoClient('mongo', 27017)
db = client.test_database
collection = db.test_collection


@app.get("/")
async def root():
    return {"message": "Hello World"}


"""@app.get("/add/{fruit}")
async def add_fruit(fruit: str):
    id = collection.insert_one({"fruit": fruit}).inserted_id 
    return {"id": str(id)}

@app.get("/list")
async def list_fruits():
    return {"results": list(collection.find({}, {"_id": False}))}
"""


class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()
species = ['setosa', 'versicolor', 'virginica']

@app.post("/predict/")
def predict(item: Item):
    item_data = jsonable_encoder(item)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        
    features = [list(item_data.values())]
    pred = model.predict(features)[0]
    specy = species[pred]
    image_path = os.path.join("images", f"{specy}.jpg")
    print("#########")
    print(image_path)

    with open(image_path, "rb") as imagefile:
        image_base64 = base64.b64encode(imagefile.read())
        print("#########")
        print(image_base64)
        
        
    return {"image": image_base64, "prediction": specy}