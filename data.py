# Author: Sriranjani Sriram

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import SimpleITK as sitk
import numpy as np

class SyntheticDatasetFromImage(Dataset):
    """Dataset for loading synthetic MRI data from image files."""

    def __init__(self, df, target_size=None):
        """Initializes the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing 'noisy_mri_dcm' and 'masks' columns.
            target_size (tuple, optional): Desired target size for resizing.
        """

        self.images = df["noisy_mri_dcm"].values
        self.masks = df["masks"].values
        self.transforms = transforms.ToTensor()
        self.target_size = target_size

    def __len__(self):
        """Returns the number of samples in the dataset."""

        return len(self.images)

    def __getitem__(self, index):
        """Returns the image and mask at the given index."""

        image_file_reader = sitk.ImageFileReader()
        image_file_reader.SetImageIO("GDCMImageIO")
        image_file_reader.SetFileName(self.images[index])
        image_file_reader.ReadImageInformation()
        image = image_file_reader.Execute()
        image = sitk.GetArrayFromImage(image)
        image = np.squeeze(image)
        image = image / np.amax(image)  # Normalize
        image = self.transforms(image).type(torch.float32)

        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = self.transforms(mask).type(torch.float32)

        return image, mask
    
class SyntheticDatasetFromNumpy(Dataset):
    """Dataset for loading synthetic MRI data from NumPy arrays."""

    def __init__(self, images, masks):
        """Initializes the dataset.

        Args:
            images (np.ndarray): NumPy array containing images.
            masks (np.ndarray): NumPy array containing masks.
        """

        self.images = images
        self.masks = masks
        self.transforms = transforms.ToTensor()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """Returns the image and mask at the given index."""

        image = self.images[index]
        image = np.squeeze(image)
        image = image / np.amax(image)  # Normalize
        image = self.transforms(image).type(torch.float32)

        mask = self.masks[index]
        mask = self.transforms(mask).type(torch.float32)

        return image, mask