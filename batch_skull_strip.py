import os
import cv2
import pydicom
import numpy as np
from tqdm import tqdm

def apply_skull_stripping(img_array):
    """
    Isolates the brain tissue and removes the skull/background noise.
    """
    img_uint8 = cv2.normalize(img_array, None,  255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_uint8 
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = np.zeros_like(img_uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    skull_stripped_img = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
    
    return skull_stripped_img

def process_dataset(input_dir, output_dir):
    """
    Walks through the dataset, applies skull stripping, and saves as PNG.
    """
    # Create the base output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gather all DICOM files first to setup the progress bar
    dcm_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dcm_files.append(os.path.join(root, file))
                
    print(f"Found {len(dcm_files)} DICOM files. Starting batch processing...")

    # Process files with a progress bar
    for file_path in tqdm(dcm_files, desc="Stripping Skulls"):
        try:
            # 1. Read DICOM
            dcm = pydicom.dcmread(file_path)
            pixel_data = dcm.pixel_array.astype(np.float32)
            
            # 2. Apply Skull Stripping
            stripped_img = apply_skull_stripping(pixel_data)
            
            # 3. Determine new save path
            # Get the relative path from the input directory
            rel_path = os.path.relpath(file_path, input_dir)
            
            # Change the extension from .dcm to .png
            rel_path_png = os.path.splitext(rel_path)[0] + ".png"
            
            # Create the absolute output path
            output_path = os.path.join(output_dir, rel_path_png)
            
            # Ensure the output subdirectories exist (e.g., NC/23f/)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 4. Save the image
            cv2.imwrite(output_path, stripped_img)
            
        except Exception as e:
            # If a file is corrupted or missing pixel data, skip it and print the error
            print(f"\nFailed to process {file_path}: {e}")

if __name__ == "__main__":
    # The folder containing your original DICOM dataset
    INPUT_DATASET_DIR = r"D:\Final Year Project\Data\Alzheimers disease MRI images"
    
    # The new folder where the stripped PNG images will be saved
    OUTPUT_DATASET_DIR = r"D:\Final Year Project\Data\test strip"
    
    process_dataset(INPUT_DATASET_DIR, OUTPUT_DATASET_DIR)
    
    print("\nBatch processing complete! Your stripped dataset is ready.")