import os
import sys
import json
import faiss
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# Function to extract feature of a single image
def extract_features(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    # add batch dimension
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()


# Step 1: Load image embeddings from the JSON file
def load_image_embeddings(embeddings_file):
    with open(embeddings_file, 'r') as f:
        image_embeddings = json.load(f)
    return image_embeddings


# Step 2: Create a FAISS index and add the image embeddings to it
def create_faiss_index(image_embeddings):
    # Dimension of embeddings
    embedding_dim = len(next(iter(image_embeddings.values())))
    embeddings = np.array(list(image_embeddings.values())).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index, list(image_embeddings.keys())


# Step 3: Define the feature extraction model and transformations
def load_feature_extractor(model_path):
    weights = ResNet50_Weights.IMAGENET1K_V1
    feature_extractor_model = models.resnet50(weights=weights)
    num_features = feature_extractor_model.fc.in_features
    feature_extractor_model.fc = torch.nn.Linear(num_features, 1000)
    feature_extractor_model.load_state_dict(torch.load(model_path))
    # Convert model to feature extractor
    feature_extractor_model.fc = torch.nn.Identity()
    feature_extractor_model.eval()
    return feature_extractor_model


# Define transformations
def get_transform():
    return transforms.Compose([
        # transforms.Resize((224, 224))
        # Convert images to tensors
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Step 4: Perform vector search to find similar images
def find_similar_images(image_path, model, transform, index, image_ids, k=5):
    query_embedding = extract_features(image_path, model, transform)
    distances, indices = index.search(np.array([query_embedding]), k)
    similar_image_ids = [image_ids[idx] for idx in indices[0]]
    return similar_image_ids


def main():
    # run from command prompt:
    # python faiss_search.py path/to/query_image.jpg

    if len(sys.argv) != 2:
        print("Usage: python faiss_search.py <query_image_path>")
        sys.exit(1)

    query_image_path = sys.argv[1]

    if not os.path.exists(query_image_path):
        print(f"Error: Image file '{query_image_path}' does not exist.")
        sys.exit(1)

    # declare paths
    embeddings_file = 'data/output/image_embeddings.json'
    model_path = 'data/final_model/image_model.pt'

    # Load the necessary components
    image_embeddings = load_image_embeddings(embeddings_file)
    index, image_ids = create_faiss_index(image_embeddings)
    feature_extractor_model = load_feature_extractor(model_path)
    transform = get_transform()

    # Find similar images
    similar_images = find_similar_images(
        query_image_path,
        feature_extractor_model,
        transform,
        index,
        image_ids,
        k=5
    )

    print(f"Similar images to '{query_image_path}': {similar_images}")


if __name__ == "__main__":
    main()



