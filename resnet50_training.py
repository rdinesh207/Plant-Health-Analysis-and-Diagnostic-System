import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets, models
import numpy as np
from torch.utils.data import random_split
from transformers import AdamW
import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "Path to dataset"


def find_images(directory):
    image_files = []
    print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                # Attempt to open the file as an image
                with Image.open(filepath) as img:
                    # If successful, add the file path to the list of image files
                    image_files.append(filepath)
            except (IOError, OSError):
                # If the file cannot be opened as an image, ignore it
                pass
    print(len(image_files))
    return image_files
    

class CropDiseaseDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # self.split = split
        self.classes = [d for d in os.listdir(self.dataset_path)]
        self.image_paths = []
        self.labels = []

        # Collect image paths and corresponding labels
        for cl in self.classes:
            data_dir = find_images(os.path.join(self.dataset_path,cl))
            labels = [self.classes.index(cl)]*len(data_dir)
            self.image_paths+=data_dir
            self.labels+=labels

    def __len__(self):
        return len(self.image_paths)

    def all_classes(self):
        return self.classes

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess the image (replace with your specific preprocessing steps)
        image = Image.open(image_path).convert('RGB')  # Assuming RGB images
        image = image.resize((224, 224))  # Resize to specific dimension (adjust as needed)
        image = torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).float()  # Convert to tensor and normalize
        #image = vis_processor(images=image, return_tensors="pt")['pixel_values'].reshape(3, 224, 224)
        image = image.to(device)
        label = torch.tensor(label).to(device)

        return image, label



dataset = CropDiseaseDataset(dataset_path)
train_dataset, val_dataset = random_split(dataset,[0.9,0.1])
train_dataloader  = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader  = DataLoader(val_dataset, batch_size=1)


num_classes=len(dataset.all_classes())
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=num_classes)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)


num_epochs = 5  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    counter = 0
    total = 0
    correct = 0
    model.train()
    train_running_loss = 0
    with tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch") as t:
        for images, labels in t:
            # Forward pass
            counter += 1
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # Use the classifier model
            loss = criterion(outputs, labels)#, alpha=0.25, gamma=2.0)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            t.set_postfix({
                  "loss": train_running_loss / counter,
                  "accuracy": correct / total,  # Only show accuracy if calculated
                  "predictions": str(predicted),
              })
        training_accuracy = correct / total
        epoch_loss = train_running_loss / counter
    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Training Accuracy: {training_accuracy * 100:.2f}%")

    model.eval()
    with torch.no_grad():  # Disable gradients for validation
        val_loss = 0.0
        val_correct = 0
        with tqdm.tqdm(val_dataloader, unit="batch") as t:
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)#, alpha=0.25, gamma=2.0)
                val_loss += loss.item()  # Accumulate validation loss

                # Calculate accuracy (assuming classification task)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_dataloader)  # Average validation loss
            val_accuracy = 100 * val_correct / len(val_dataset)  # Validation accuracy


    # Print training and validation results
    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Training Accuracy: {training_accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Save the trained model (optional)
    torch.save(model.state_dict(), "resnet50.pth")

