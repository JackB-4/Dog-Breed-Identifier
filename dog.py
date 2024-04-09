import os, torch, json
from torch.utils.data import DataLoader
from training_functions import model_prep, train_model, evaluate_model, test_accuracy, DogBreedDataset, transform

# Load the settings
with open('config.json', 'r') as f:
    config = json.load(f)

# Apply the settings
TRAIN_MODE = config['TRAIN_MODE']
ACCURACY_MODE = config['ACCURACY_MODE']
MODEL_PATH = config['MODEL_PATH']
test_dir = config['test_dir']
train_dir = config['train_dir']
batch_size = config['batch_size']
num_epochs = config['num_epochs']


#Check if using GPU
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current macOS version is not 12.3+ or you do not have an MPS-enabled device on this machine.")
else:
    print("MPS is available!")


def main():
    model, criterion, optimizer, scheduler, train_loader, val_loader, device = model_prep(train_dir, batch_size)
    if ACCURACY_MODE:

        model.load_state_dict(torch.load(MODEL_PATH))
        test_accuracy(model, val_loader, device)

    elif TRAIN_MODE:

        # Save the breed labels to a file
        breed_dataset = DogBreedDataset(train_dir, transform=transform)
        breed_labels = breed_dataset.breed_labels
        with open('breed_labels.json', 'w') as f:
            json.dump(breed_labels, f)
    
        train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs)

    elif not TRAIN_MODE:

        #Loads the breed labels
        eval_dataset = DogBreedDataset(test_dir, transform=transform, include_labels=False)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        with open('breed_labels.json', 'r') as f:
            breed_labels = json.load(f)
        index_to_breed = {v: k for k, v in breed_labels.items()}

        #Loads the saved model and evaluates it on test data
        model.load_state_dict(torch.load(MODEL_PATH))
        evaluate_model(model, eval_loader, criterion, device, index_to_breed, include_labels=False)

if __name__ == '__main__':
    main()