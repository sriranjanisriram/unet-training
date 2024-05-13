# Author: Sriranjani Sriram

"""
This script processes a set of DICOM images, applies noise, 
creates corresponding masks, and saves the results in various formats.

The script supports saving data as individual image files (PNG and DCM)
and as NumPy arrays.
"""

from pathlib import Path
import os
import copy
import multiprocessing
import functools
import itertools
import gc

import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd


def rescale_intensity(dcm, image_file_reader):
    """
    Rescales the intensity of a DICOM image to 0-255 and converts it to 8-bit unsigned integer.

    Args:
    - dcm: SimpleITK image object.
    - image_file_reader: SimpleITK image file reader object used to read the DICOM.

    Returns:
    - SimpleITK image object with rescaled intensity.
    """

    if dcm.GetNumberOfComponentsPerPixel() == 1:
        dcm = sitk.RescaleIntensity(dcm, 0, 255)
        if image_file_reader.GetMetaData("0028|0004").strip() == "MONOCHROME1":
            dcm = sitk.InvertIntensity(dcm, maximum=255)
        dcm = sitk.Cast(dcm, sitk.sitkUInt8)
    
    return dcm


def zero_pad_to_fixed_size(array, target_shape):
    """
    Zero-pads the input array to the target shape.

    Args:
    - array: Input NumPy array.
    - target_shape: Tuple specifying the target shape (height, width) after padding.

    Returns:
    - Padded array.
    """

    # Calculate padding for each dimension
    pad_height = max(0, target_shape[0] - array.shape[0])
    pad_width = max(0, target_shape[1] - array.shape[1])

    # Calculate padding tuple
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Perform zero-padding
    padded_array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return padded_array


def zero_pad_dicom_image(image, target_size):
    """
    Pads a DICOM image with a constant value using SimpleITK.

    Args:
    - image: SimpleITK image object.
    - target_size: Tuple specifying the target size (height, width) after padding.

    Returns:
    - Padded SimpleITK image object.
    """

    # Get original size
    original_size = image.GetSize()

    # Calculate padding for each dimension
    pad_size = [(target - original) // 2 for target, original in zip(target_size, original_size)]

    # Create padding filter and set parameters
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound(pad_size)
    pad_filter.SetPadUpperBound(pad_size)
    pad_filter.SetConstant(0)

    # Apply padding
    padded_image = pad_filter.Execute(image)

    return padded_image

def add_salt_and_pepper_noise(image, salt_value=255, pepper_value=0, probability=0.1):
    """
    Adds salt-and-pepper noise to a SimpleITK image.

    Args:
    - image: SimpleITK image object.
    - salt_value: Value for salt noise.
    - pepper_value: Value for pepper noise.
    - probability: Probability of noise for each pixel.

    Returns:
    - SimpleITK image object with added noise.
    """

    # Convert to NumPy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Add noise
    U = np.random.rand(*image_array.shape)
    salt_value = np.random.randint(low=0, high=salt_value, size=image_array.shape)
    salt_pixels = U < probability/2
    pepper_pixels = U > 1- probability/2
    image_array[salt_pixels] = salt_value[salt_pixels]
    image_array[pepper_pixels] = pepper_value
    
    # Convert back to SimpleITK image
    noisy_image = sitk.GetImageFromArray(image_array)
    noisy_image.CopyInformation(image)
    
    return noisy_image


def create_mask(dcm_file, iterate, dataset_folder, image_size=None, return_np=False):
    """
    Creates a mask for a DICOM image, adds noise, and saves the results in various formats.

    Args:
    - dcm_file: Path to the DICOM file.
    - iterate: Iteration number (used for file naming).
    - dataset_folder: Path to the dataset folder.
    - image_size: Target size for resizing (optional).
    - return_np: Whether to return the noisy image and mask as NumPy arrays.

    Returns:
    - If return_np is True: tuple of (noisy_image_npy, mask)
    - Otherwise: None
    """

    # Read DICOM file
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO("GDCMImageIO")
    image_file_reader.SetFileName(dcm_file)
    image_file_reader.ReadImageInformation()

    dcm = image_file_reader.Execute()
    dcm = dcm[:, :, 0]

    # Resize image if target size is specified
    if image_size:
        original_size = dcm.GetSize()
        original_spacing = dcm.GetSpacing()
        new_spacing = [
            (original_size[0] - 1) * original_spacing[0] / (image_size - 1),
            (original_size[1] - 1) * original_spacing[1] / (image_size - 1)
        ]
        new_size = [
            image_size,
            image_size,
        ]
        dcm = sitk.Resample(
            image1=dcm,
            size=new_size,
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            outputOrigin=dcm.GetOrigin(),
            outputSpacing=new_spacing,
            outputDirection=dcm.GetDirection(),
            defaultPixelValue=0,
            outputPixelType=dcm.GetPixelID(),
        )
    
    # Create writer for saving data
    writer = sitk.ImageFileWriter()

    # Save original DICOM
    writer.SetFileName(dataset_folder + f"mri_dcm/{iterate}.dcm")
    writer.Execute(dcm)

    # Rescale intensity for masking
    dcm_rs = rescale_intensity(dcm, image_file_reader)

    # Save original PNG
    writer.SetFileName(dataset_folder + f"mri_png/{iterate}.png")
    writer.Execute(dcm_rs)
    
    # Create mask using OpenCV
    numpy_data = sitk.GetArrayFromImage(dcm_rs)
    numpy_data = np.squeeze(numpy_data)
    mask = cv2.threshold(numpy_data, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Zero-pad mask to 512x512 if target size is not specified
    if not image_size:
        mask = zero_pad_to_fixed_size(mask, (512, 512))

    # Save mask as PNG
    cv2.imwrite(dataset_folder + f"masks/{iterate}.png", mask)
    
    # Add Gaussian blur noise
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(float(2.0))
    blur_image = gaussian.Execute(dcm)

    # Add salt-and-pepper noise
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(dcm)
    max_pixel_value = stats_filter.GetMaximum() # Get maximum pixel value for noise
    blur_image = add_salt_and_pepper_noise(blur_image, max_pixel_value, 0, 0.01)

    # Cast image to original pixel type
    pixelID = dcm.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    blur_image = caster.Execute(blur_image)

    # Zero-pad noisy image to 512x512 if target size is not specified
    if not image_size:
        blur_image = zero_pad_dicom_image(blur_image, (512, 512))

    # Save noisy DICOM
    writer.SetFileName(dataset_folder + f"noisy_mri_dcm/{iterate}.dcm")
    writer.Execute(blur_image)
    
    # Convert noisy image to NumPy array
    blur_image_npy = sitk.GetArrayFromImage(blur_image)

    # Zero-pad noisy image array to 512x512 if target size is not specified
    if not image_size:
        blur_image_npy = zero_pad_to_fixed_size(blur_image_npy, (512, 512))
        
    # Rescale intensity for saving noisy PNG
    blur_image_rs = rescale_intensity(blur_image, image_file_reader)

    # Save noisy PNG
    if not image_size:
        blur_image_rs = sitk.GetArrayFromImage(blur_image_rs)
        blur_image_rs = zero_pad_to_fixed_size(blur_image_rs, (512, 512))
        cv2.imwrite(dataset_folder + f"noisy_mri_png/{iterate}.png", blur_image_rs)
    else:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(dataset_folder + f"noisy_mri_png/{iterate}.png")
        writer.Execute(blur_image_rs)

    if return_np:
        return blur_image_npy, mask
    

def convert_images(dcm_files, index_range, dataset_folder, image_size):
    """
    Processes a list of DICOM files using multiprocessing to create masks and add noise.

    Args:
    - dcm_files: List of paths to DICOM files.
    - index_range: List of indices for file naming.
    - dataset_folder: Path to the dataset folder.
    - image_size: Target size for resizing (optional).
    """

    MAX_PROCESSES = 15
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        pool.starmap(
            functools.partial(create_mask, dataset_folder=dataset_folder, image_size=image_size, return_np=False),
            zip(dcm_files, index_range),
        )   


def main():
    """
    Main function to process DICOM images, create masks, add noise, and save results.
    """

    dataset_folder = "Dataset/"
    save_numpy = True
    image_size = None

    # Create necessary directories
    os.makedirs(dataset_folder + "mri_dcm", exist_ok=True)
    os.makedirs(dataset_folder + "mri_png", exist_ok=True)
    os.makedirs(dataset_folder + "noisy_mri_dcm", exist_ok=True)
    os.makedirs(dataset_folder + "noisy_mri_png", exist_ok=True)
    os.makedirs(dataset_folder + "masks", exist_ok=True)

    # Find all DICOM files
    result = list(Path("mri_data").rglob("*.[dD][cC][mM]"))
    mri_original = []
    masks_path = []
    mri_dcm_path = []
    mri_png_path = []
    noisy_dcm_path = []
    noisy_png_path = []

    iterate = 0
    for dcm in result:
        dcm = str(dcm)
        file_name = dcm.split("/")[-1]
        if file_name[0] != ".":
            
            mri_original.append(dcm)
            mri_dcm_path.append(dataset_folder + f"mri_dcm/{iterate}.dcm")
            mri_png_path.append(dataset_folder + f"mri_png/{iterate}.png")
            masks_path.append(dataset_folder + f"masks/{iterate}.png")
            noisy_dcm_path.append(dataset_folder + f"noisy_mri_dcm/{iterate}.dcm")
            noisy_png_path.append(dataset_folder + f"noisy_mri_png/{iterate}.png")
            iterate += 1

    if not save_numpy:
        index_range = np.arange(iterate).tolist()
        convert_images(mri_original, index_range, dataset_folder, image_size)
    else:
        # Create NumPy arrays for noisy images and masks
        if image_size:
            noisy_np = np.empty((len(mri_original), image_size, image_size), dtype=np.float16)
            masks_np = np.empty((len(mri_original), image_size, image_size), dtype=np.uint8)
        else:
            noisy_np = np.empty((len(mri_original), 512, 512), dtype=np.float16)
            masks_np = np.empty((len(mri_original), 512, 512), dtype=np.uint8)
        noisy_np.fill(0)
        masks_np.fill(0)

        # Process each DICOM file and populate NumPy arrays
        for i in range(iterate):
            print(f"{i}/{iterate}")
            noisy, mask = create_mask(mri_original[i], i, dataset_folder, image_size, return_np=True)
            noisy_np[i, :, :] = copy.deepcopy(noisy)
            masks_np[i, :, :] = copy.deepcopy(mask)
            
        # Save NumPy arrays
        np.save(dataset_folder + "noisy_mri.npy", noisy_np)
        np.save(dataset_folder + "masks.npy", masks_np)

    # Create and save dataset CSV file
    pd.DataFrame({
        "mri_dcm": mri_dcm_path,
        "mri_png": mri_png_path,
        "masks": masks_path,
        "noisy_mri_dcm": noisy_dcm_path,
        "noisy_mri_png": noisy_png_path
    }).to_csv(dataset_folder + "dataset.csv", index=False)

if __name__ == "__main__":
    main()