import os, torch, json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Load the settings
with open('config.json', 'r') as f:
    config = json.load(f)

# Apply the settings
TRAIN_MODE = config['TRAIN_MODE']
test_dir = config['test_dir']
train_dir = config['train_dir']
batch_size = config['batch_size']
num_epochs = config['num_epochs']



class DogBreedDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_labels=True):
        """
        Args:
            root_dir (string): Directory with all the images or breed subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
            include_labels (bool, optional): Indicates if the labels are included (True) or not (False).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.include_labels = include_labels
        self.images = []
        self.labels = []
        self.breed_labels = {}

        if include_labels:
            # Original logic for loading with labels
            breed_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and '-' in d]
            breed_index = 0
            for breed_dir in breed_dirs:
                breed_name = breed_dir.split('-', 1)[1]
                if breed_name not in self.breed_labels:
                    self.breed_labels[breed_name] = breed_index
                    breed_index += 1
                
                for img_name in os.listdir(os.path.join(root_dir, breed_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        img_path = os.path.join(root_dir, breed_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.breed_labels[breed_name])
        else:
            # Logic for loading without labels
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(root_dir, img_name)
                    self.images.append(img_path)
                    # No labels are appended here, -1 is used as a placeholder
                    self.labels.append(-1)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Return image and label or image only based on include_labels
        return (image, self.labels[idx]) if self.include_labels else image



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prints the top 3 predictions for each image in the batch
def print_topk_predictions(outputs, index_to_breed, topk):
    probabilities, indices = torch.topk(F.softmax(outputs, dim=1), topk, dim=1)
    probabilities = probabilities.cpu()
    indices = indices.cpu()
    for i in range(outputs.size(0)):  # Loop through batch
        print(f"Image {i+1}:")
        for j in range(topk):
            breed = index_to_breed[int(indices[i][j])]
            probability = probabilities[i][j].item()
            probability = probability * 100
            print(f"  {breed}: {probability:3.1f}%")


def evaluate_model(model, dataloader, criterion, device, index_to_breed, include_labels=True, topk=3):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    
    with torch.no_grad():
        for data in dataloader:
            if include_labels:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)
            else:
                images = data.to(device)
                outputs = model(images)
                # Print top-k predictions for non-labeled data
                print_topk_predictions(outputs, index_to_breed, topk)

    if include_labels:
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_images
        print(f'Loss: {avg_loss:.4f}, Accuracy: {(100 * accuracy):.2f}%')
        return avg_loss, accuracy
    else:
        print("Evaluation completed without labels.")
        return None




def model_prep(data_dir, batch_size=32):
    dataset = DogBreedDataset(root_dir=data_dir, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)


    # Define GPU or CPU
    mps_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    if TRAIN_MODE:
        # Load a pre-trained model and move it to GPU
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(mps_device)

        # Replace the final fully connected layer with one that matches the number of breeds
        num_breeds = len(dataset.breed_labels)
        model.fc = nn.Linear(model.fc.in_features, num_breeds)

    else:
        model = models.resnet50(weights=None)  # Initialize model
        num_breeds = len(dataset.breed_labels)
        model.fc = nn.Linear(model.fc.in_features, num_breeds)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    return model, criterion, optimizer, scheduler, train_loader, val_loader, mps_device


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=25):
    print('Training model...')
    model.to(device)
    best_accuracy = 0

    #TRAINING PHASE
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20 == 19:  # Mid-epoch logging
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss: {running_loss / 15:.4f}')
                running_loss = 0.0

        # VALIDATION STAGE
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}: Accuracy: {accuracy:.2f}%')

        scheduler.step(val_loss)

        # Save the model if it has the best accuracy so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved new best model with accuracy: {best_accuracy:.2f}%')

    # Saves final model
    torch.save(model.state_dict(), 'final_model.pth')
    print('Saved final model.')




def test_accuracy(model, val_loader, device):
    print('Testing model accuracy...')
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Model Accuracy: {accuracy:.2f}%')