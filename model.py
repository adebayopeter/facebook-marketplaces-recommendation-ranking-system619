import time

import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import pickle
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter


class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, file_extension = '.jpg'):
        """
        Args:
            dataframe (DataFrame): Pandas DataFrame with names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.file_extension = file_extension

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.dataframe.iloc[idx, 0]) + self.file_extension)
        print(f"Loading image: {img_name}")
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError as e:
            print(f'File not found: {img_name}')
            raise e
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Create function for model folder
def create_model_dir(base_dir='data/model_evaluation'):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_dir = os.path.join(base_dir, f'model_{timestamp}')
    weights_dir = os.path.join(model_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    return model_dir, weights_dir


if __name__ == "__main__":
    # Load CSV file
    dataframe = pd.read_csv('data/csv/training_data.csv', dtype={'image_id': str, 'category_label': int}, index_col=0)

    # Create the encoder and decoder dictionaries
    labels = dataframe['category_label'].unique()
    label_encoder = {label: idx for idx, label in enumerate(labels)}
    label_decoder = {idx: label for label, idx in label_encoder.items()}

    # Save the decoder dictionary
    with open('data/resource/image_decoder.pkl', 'wb') as f:
        pickle.dumps(label_decoder)

    # Split dataset into training, validation and test sets
    df_training, df_temp = train_test_split(
        dataframe, test_size=0.4, stratify=dataframe['category_label'])
    df_validation, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=df_temp['category_label']
    )

    # Define transformations
    transform = transforms.Compose([
        # transforms.Resize((224, 224))
        # Convert images to tensors
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset instance
    train_dataset = CustomDataset(
        dataframe=df_training,
        root_dir='data/clean_images/',
        transform=transform,
        file_extension='.jpg'
    )
    validation_dataset = CustomDataset(
        dataframe=df_validation,
        root_dir='data/clean_images/',
        transform=transform,
        file_extension='.jpg'
    )
    test_dataset = CustomDataset(
        dataframe=df_test,
        root_dir='data/clean_images/',
        transform=transform,
        file_extension='.jpg'
    )

    # Create dataloader instance
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Load pre-trained ResNet-50 model
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(label_encoder))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize Tensorboard writer
    writer = SummaryWriter('data/resource/tensorboard')
    # To run tensorboard: tensorboard --logdir=data/resource

    # Create the directories to save model weights & metrics
    model_dir, weights_dir = create_model_dir()

    # Training function
    def train(model, epochs=1):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (images, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Logs every 10 batches
                if i % 10:
                    # logs the average training loss for the last 10 batches
                    writer.add_scalar(
                        'training loss',
                        running_loss / 10,
                        epoch * len(train_dataloader) + i
                    )
                    running_loss = 0.0

            # validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in validation_dataloader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(validation_dataloader)
            writer.add_scalar(
                'validation loss',
                avg_val_loss,
                epoch
            )

            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

            # Save the model weights at the end of the epoch
            weights_path = os.path.join(weights_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), weights_path)

            # Save metrics e.g. loss at the end of each epoch
            metrics_path = os.path.join(model_dir, 'metrics.txt')
            with open(metrics_path, 'a') as f:
                f.write(f'Epoch {epoch + 1}, Training loss: {running_loss:.4f}, Validation loss: {avg_val_loss:.4f}\n')

        writer.flush()

    # Call the training function
    train(model, 10)
    writer.close()

    # Example usage
    # for images, labels in dataloader:
    #    print(images.size(), labels)
