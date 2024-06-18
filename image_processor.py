import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Define transformations
transform = transforms.Compose([
    # transforms.Resize((224, 224))
    # Convert images to tensors
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def process_image(image_path):
    if not os.path.exists(image_path):
        print(f'Error file image {image_path} does not exist')
        return None

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Apply transformation
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    return image


if __name__ == "__main__":
    processed_image = process_image(image_path='data/test_img/tv1.jpeg')
    print(f'Processed image shape: {processed_image.shape}')
