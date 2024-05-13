# Author: Sriranjani Sriram

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split

# Import custom modules
from data import SyntheticDatasetFromImage, SyntheticDatasetFromNumpy
from model import UNet
from utils import iou_score, EarlyStopper

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping):
    """
    Trains the U-Net model.

    Args:
        model (nn.Module): The U-Net model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for the model.
        num_epochs (int): Number of epochs to train for.
        early_stopping (EarlyStopper): Early stopping object.

    Returns:
        tuple: Training and validation loss lists.
    """
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0.0

        # Iterate over the training data
        for images, masks in tqdm(train_loader):
            # Move data to the device
            images, masks = images.to(device), masks.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the running loss
            running_loss += loss.item() * images.size(0)

        # Append the average training loss for the epoch
        train_loss.append(running_loss / len(train_loader.dataset))

        # Set the model to evaluation mode
        model.eval()
        running_loss = 0.0

        # Disable gradient calculation
        with torch.no_grad():
            # Iterate over the validation data
            for images, masks in val_loader:
                # Move data to the device
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate the loss
                loss = criterion(outputs, masks)

                # Accumulate the running loss
                running_loss += loss.item() * images.size(0)

        # Append the average validation loss for the epoch
        val_loss.append(running_loss / len(val_loader.dataset))

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

        # Check for early stopping
        if early_stopping.early_stop(val_loss[-1], model):
            break

    # Return the training and validation loss lists
    return train_loss, val_loss


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    lr = 1e-3
    epochs = 20
    early_stopping_patience = 5

    # Define paths
    plots_folder = "plots/"
    model_save_folder = "models/"
    data_folder = "Dataset/"

    # Training flags
    train_run = True
    train_from_numpy = True

    # Create necessary folders
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(model_save_folder, exist_ok=True)

    # Load data based on training flags
    if train_from_numpy and train_run:
        """
        We load the entire data as numpy array simply because it is faster and the dataset used is relatively managable (14 GB).
        """

        # Load numpy data
        dataset = np.load(data_folder + "noisy_mri.npy")
        

        # Split data into train, validation, and test sets
        index = np.arange(len(dataset))
        train_index, temp_index = train_test_split(index, test_size=0.3, random_state=42)
        val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=42)

        # Extract data subsets
        train_data = dataset[train_index]
        val_data = dataset[val_index]
        test_data = dataset[test_index]
        
        # Free up memory
        del dataset

        # Load mask data
        mask_array = np.load(data_folder + "masks.npy")
        train_mask = mask_array[train_index]
        val_mask = mask_array[val_index]
        test_mask = mask_array[test_index]
        
        del mask_array

        # Load and split CSV data
        df = pd.read_csv(data_folder + "dataset.csv")
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]
        df_val = df.loc[val_index]

        # Save each subset as a CSV file
        df_train.to_csv(data_folder + "train_data.csv", index=False)
        df_test.to_csv(data_folder + "test_data.csv", index=False)
        df_val.to_csv(data_folder + "val_data.csv", index=False)

        

        # Create PyTorch datasets
        train_dataset = SyntheticDatasetFromNumpy(train_data, train_mask)
        val_dataset = SyntheticDatasetFromNumpy(val_data, val_mask)
        test_dataset = SyntheticDatasetFromNumpy(test_data, test_mask)

    else:
        # Load CSV data
        if not os.path.isfile(data_folder + "train_data.csv"):
            df = pd.read_csv(data_folder + "dataset.csv")

            # Split data into train, validation, and test sets
            train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
            test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

            # Save the split datasets
            train_data.to_csv(data_folder + "train_data.csv", index=False)
            test_data.to_csv(data_folder + "test_data.csv", index=False)
            val_data.to_csv(data_folder + "val_data.csv", index=False)
        else:
            train_data = pd.read_csv(data_folder + "train_data.csv")
            test_data = pd.read_csv(data_folder + "test_data.csv")
            val_data = pd.read_csv(data_folder + "val_data.csv")

        # Create PyTorch datasets
        train_dataset = SyntheticDatasetFromImage(train_data)
        val_dataset = SyntheticDatasetFromImage(val_data)
        test_dataset = SyntheticDatasetFromImage(test_data)

    # Create PyTorch data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Instantiate the model, loss function, optimizer, and early stopping
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopper(patience=early_stopping_patience, folder=model_save_folder)

    # Train the model
    if train_run:
        train_loss, val_loss = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs,
            early_stopping,
        )

        # Plot training and validation loss
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Training Loss')
        ax.plot(range(1, len(train_loss) + 1), val_loss, 'r--', label='Validation Loss')

        # Enhancements
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(axis='y', alpha=0.5) 
        fig.savefig(plots_folder + f"AI-train_loss.png")

        # Free up memory
        del train_data, val_data, train_mask, val_mask
        gc.collect()

    # Load the best model parameters
    model.load_state_dict(torch.load(model_save_folder + "best-model-parameters.pt"))

    # Create a sample input for tracing and scripting
    (dummy_input, _) = next(iter(train_loader))

    # Trace and script the model for C++ deployment (if training)
    if train_run:
        traced_model = torch.jit.trace(
            model.to(torch.device("cuda")), dummy_input.to(torch.device("cuda"))
        )
        scripted_model = torch.jit.script(model)
        scripted_model.save(model_save_folder + "model_cpp.pt")

    # Set the model to evaluation mode
    model.to(device)
    model.eval()

    # Test the model
    test_loss = 0.0
    with torch.no_grad():
        # Iterate over the test data
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            # Move data to the device
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid activation and threshold
            outputs = torch.sigmoid(outputs)
            outputs = torch.where(outputs < 0.5, 0, 1)

            # Calculate IOU score and plot the first batch
            if i < 1:
                for j in range(len(outputs)):
                    loss = iou_score(outputs[j], masks[j])
                    test_loss += loss.item()

                    # Plot original and segmented masks
                    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
                    fig.suptitle(f"AI segmentation with IOU: {loss.item()}")

                    ax[0].imshow(masks[j].detach().cpu().numpy().squeeze(), cmap="gray")
                    ax[0].set_title("Original")
                    ax[0].set_axis_off()

                    ax[1].imshow(outputs[j].detach().cpu().numpy().squeeze(), cmap="gray")
                    ax[1].set_title("Segmented Mask")
                    ax[1].set_axis_off()

                    fig.tight_layout()
                    fig.savefig(plots_folder + f"AI-test_{i*batch_size+j}.png")

                    # Clear figure and release memory
                    fig.clear()
                    plt.close(fig)
                    gc.collect()

    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"IOU Score: {test_loss:.4f}")