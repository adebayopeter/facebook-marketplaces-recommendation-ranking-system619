# Facebook Marketplace's Recommendation Ranking System

Facebook marketplace recommendation ranking system project 

## Table of Contents

1. [Description](#description)
2. [Objectives](#objectives)
3. [Overview](#overview)
  - [Pre-trained ResNet-50 Model](#pre-trained-resnet-50-model)
  - [Training the Model](#training-the-model)
  - [Feature Extraction](#feature-extraction)
  - [Similarity Search with FAISS](#similarity-search-with-faiss)
4. [Goals](#goals)
5. [Conclusion](#conclusion)
6. [Installation](#installation)
7. [Usage](#usage)
8. [File Structure](#file-structure)
9. [TensorBoard Visualizations](#tensorboard-visualizations)
10. [License](#license)

## Description

The Facebook Marketplace is a platform for buying and selling products on Facebook, providing users with personalized recommendations for listings based on their search queries. This project aims to replicate the underlying system of the Marketplace, employing artificial intelligence to recommend the most relevant listings.

## Objectives

The primary objective of this project is to understand the process of fine-tuning a pre-trained model for feature extraction and using these features for similarity searches.

## Overview

### Pre-trained ResNet-50 Model

We utilize a pre-trained ResNet-50 model to extract features from images. ResNet-50 is a powerful convolutional neural network (CNN) that has been trained on a large dataset (ImageNet) and is capable of capturing rich feature representations from images.

### Training the Model

Initially, a classification model is trained using a custom dataset. This involves:
1. Loading the pre-trained ResNet-50 model.
2. Fine-tuning the model by freezing most layers and retraining the last few layers on our dataset.
3. Training the modified model to classify images based on the custom dataset.

### Feature Extraction

After training the classification model, it is converted into a feature extraction model by:
1. Removing the last few fully connected layers.
2. Adjusting the final layer to output a fixed-size feature vector (1000 dimensions).

### Similarity Search with FAISS

The extracted features (embeddings) are then used to perform similarity searches. This involves:
1. Creating a dictionary of image IDs and their corresponding embeddings.
2. Using Facebook AI Similarity Search (FAISS) to efficiently search and find similar images based on their embeddings.

## Goals

- Build a robust feature extraction model from a pre-trained ResNet-50.
- Extract image embeddings to represent the content of images.
- Implement FAISS for efficient similarity searches on these embeddings.
- Provide personalized and relevant product recommendations based on similarity searches.

## Conclusion

By the end of this project, we aim to have a functional system that mirrors the recommendation engine of the Facebook Marketplace, leveraging deep learning and similarity search techniques to deliver personalized and relevant product recommendations to users.

## Usage Instructions

1. Train the model by running the training script:
    ```sh
    python train_model.py
    ```

2. Extract features from images:
    ```sh
    python extract_features.py
    ```

3. Perform similarity search using FAISS:
    ```sh
    python similarity_search.py <image_path>
    ```

## Installation

### Prerequisites
- Python 3.6+
- PyTorch
- TorchVision
- Pandas
- scikit-learn
- Pillow
- TensorBoard

### Instructions
1. Clone the repository:
```bash
git clone https://github.com/adebayopeter/facebook-marketplaces-recommendation-ranking-system619.git
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate # On windows use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create data resource folders/files
Create a folder `data` that would serve as our main resource folder.
Inside `data` create a folder `csv` that holds the csv dataset of `Image.csv`, `Prodcts.csv`, `merged_data.csv` and `cleaned_images.csv`. 
Inside the folder, create `images` folder that holds the image dataset and `clean_images` folder where we store the cleaned images.

## File Structure
The project directory is structured as follows:

```
📦 facebook_markeplace
├─ data
│  ├─ csv
│  │  ├─ Products.csv
│  │  ├─ training_data.csv
│  │  ├─ merged_data.csv
│  │  └─ cleaned_images.csv
│  ├─ clean_images
│  ├─ images
│  ├─ model_evaluation
│  └─ resource
│     ├─ tensorboard
│     └─ image_decorder.pkl
├─ src
├─ .gitignore
├─ clean_images.py
├─ clean_tabular_data.py
├─ faiss_search.py
├─ image_processor.py
├─ main.py
├─ model.py
├─ README.md
└─ requirements.txt
```

## TensorBoard Visualizations

### Training and Validation Loss
![Training and Validation Loss](src/tensorboard.png)

## License
This project is licensed under [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)