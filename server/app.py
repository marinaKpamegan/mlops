from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pickle
from io import BytesIO
import base64, json
from PIL import Image
import os
import mlflow
from train import *
from sklearn.datasets import load_iris


app = FastAPI()
client = MongoClient('mongo', 27017)
db = client.test_database
collection = db.test_collection
current_model = None


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
model_files = {"KNN":"knn", "Random Forest Classifier":"random_forest", "Decision Tree Classifier":"decision_tree"}

# Chargement des données Iris
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)


@app.post("/predict/{model_name}")
def predict(item: Item, model_name="KNN", model_version=None):
    chosen_model = model_files[model_name]
    item_data = jsonable_encoder(item)
    print("#############")
    print(chosen_model)
    # load model from mlflow
    model = get_model_from_mlflow(chosen_model, model_version)
    # with open(model_path, "rb") as f:
    #     model_file = pickle.load(f)
       
    features = [list(item_data.values())]
    input_array = np.array(features).reshape(1, -1)
    print("#############")
    print(input_array)
    
    pred = model.predict(input_array)[0]
    specy = species[pred]
    image_path = os.path.join("images", f"{specy}.jpg")

    with open(image_path, "rb") as imagefile:
        image_base64 = base64.b64encode(imagefile.read())       
        
    return {"image": image_base64, "prediction": specy}


def get_model_from_mlflow(model_name:str, model_version:int = None):
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    print("#############")
    print(model_uri)
    model = mlflow.pyfunc.load_model(
        model_uri=model_uri
    )
    return model

@app.post("/train/{model_name}/{test_size}")
def train(model_name: str, test_size: float = 0.4):
    try:
        runned_model, matrix = run_model(iris.data, iris.target, iris.target_names, model_type=model_name, test_size=test_size)
        return {"success": True, "message": f"Model {model_name} trained successfully", "model_link":f"http://localhost:5000/#/models/{runned_model}", "matrix_base64": matrix}
    except Exception as e:
        # Renvoie une erreur HTTP avec un code 400 (bad request)
        raise HTTPException(status_code=400, detail=str(e))

    
@app.get("/update-model")
def update_model(model_name: str, model_version: int = None):
    global current_model
    try:
        """ if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/latest" """

        # Load the model from MLFlow
        current_model = get_model_from_mlflow(model_name=model_name)
        return {"success": True, "message": f"Model '{model_name}' (version {model_version or 'latest'}) loaded successfully."}
    except Exception as e:
        return {"success": False, "error": str(e)}
