import pickle

import faiss
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from torchvision import transforms, models
import json
import numpy as np

# Import your image processing script here
from image_processor import process_image


class FeatureExtractor(nn.Module):
    def __init__(self, decoder: dict = None):
        super(FeatureExtractor, self).__init__()

        # Load pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        # Adjust base on model's output size
        self.model.fc = nn.Linear(num_features, 1000)

        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x


# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
    # Load the Feature Extraction model
    model_weights_path = 'data/final_model/image_model.pt'
    feature_extractor_model = FeatureExtractor()
    feature_extractor_model.load_state_dict(torch.load(model_weights_path))
    feature_extractor_model.eval()

except:
    raise OSError("No Feature Extraction model found. "
                  "Check that you have the decoder and the model in the correct location")

try:
    # Load the FAISS model. Use this space to load the FAISS model   #
    # which was saved as a pickle with all the image embeddings   #
    # fit into it.
    with open('data/output/image_embeddings.json', 'r') as f:
        image_embeddings = json.load(f)

    embedding_matrix = np.array(list(image_embeddings.values()))
    image_ids = list(image_embeddings.keys())
    faiss_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    faiss_index.add(embedding_matrix)

except:
    raise OSError("No Image model found. Check that you have the encoder "
                  "and the model in the correct location")

app = FastAPI()
print("Starting server")


@app.get('/healthcheck')
def healthcheck():
    msg = "API is up and running!"
    return {"message": msg}


@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    # Process the input and use it as input for the feature extraction model image.
    image_tensor = process_image(pil_image)
    features = feature_extractor_model.predict(image_tensor).numpy()

    # Return the image embeddings here
    return JSONResponse(content={
        "features": features.tolist(),
    })


@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)

    #####################################################################
    # Process the input  and use it as input for the feature            #
    # extraction model.File is the image that the user sent to your API #   
    # Once you have feature embeddings from the model, use that to get  # 
    # similar images by passing the feature embeddings into FAISS       #
    # model. This will give you index of similar images.                #            
    #####################################################################
    image_tensor = process_image(pil_image)
    features = feature_extractor_model.predict(image_tensor).numpy()

    # Get similar images using FAISS
    _, indices = faiss_index.search(features, k=10)  # Get top 10 similar images

    result_image_ids = [image_ids[idx] for idx in indices[0]]
    decoded_labels = [feature_extractor_model.decoder[int(img_id)] for img_id in result_image_ids]

    return JSONResponse(content={
        "similar_index": result_image_ids,  # Return the index of similar images here
        "labels": decoded_labels
    })


if __name__ == '__main__':
    uvicorn.run("api_template:app", host="0.0.0.0", port=8080)
