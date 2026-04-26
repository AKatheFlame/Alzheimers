"""
Medical Image Preprocessing Pipeline
Author: Akarsh
"""
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys

def apply_skull_stripping(img_array):
    # Normalize to 0-255 uint8 for OpenCV processing
    img_uint8 = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Otsu's Thresholding
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological opening to detach the skull from the brain
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_uint8 
        
    # Assume the largest contour is the brain
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask and extract brain
    mask = np.zeros_like(img_uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    skull_stripped_img = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
    
    return img_uint8, skull_stripped_img

if __name__ == "__main__":
    # Replace with the path to a valid DICOM file from your dataset
    DICOM_PATH = r"D:\Final Year Project\Data\Alzheimers disease MRI images\NC\24f\IM00011.dcm"
    
    try:
        dcm = pydicom.dcmread(DICOM_PATH)
        pixel_data = dcm.pixel_array.astype(np.float32)
    except Exception as e:
        print(f"Error loading DICOM file: {e}")
        print("Please ensure 'sample_mri.dcm' exists in the same folder.")
        sys.exit()

    original_img, stripped_img = apply_skull_stripping(pixel_data)

    # Visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original MRI Slice")
    plt.imshow(original_img, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Skull-Stripped Brain")
    plt.imshow(stripped_img, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()