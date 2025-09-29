import numpy as np
from skimage import color, img_as_ubyte

def dental_gray_world_white_balance(image_rgb):
    """
    Apply modified gray-world white balance for dental images.
    Preserves red (gums/tongue) while balancing white (teeth).
    """
    img_float = image_rgb.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Convert to HSV to detect white (teeth) and red (gums/tongue)
    img_hsv = color.rgb2hsv(img_float)

    # Create a "teeth mask" (high brightness)
    # teeth_mask = img_hsv[..., 2] > 0.75  # V (brightness) threshold for teeth

    # Create a "red mask" (gums/tongue)
    # red_mask = ((img_hsv[..., 0] > 0.95) | (img_hsv[..., 0] < 0.05)) & (img_hsv[..., 1] > 0.4)  # H (hue) for red
    blue_mask = ((img_hsv[..., 0] > 0.43) & (img_hsv[..., 0] < 0.70)) & ((img_hsv[..., 1] > 0.30) &((img_hsv[..., 1] < 0.55))) #this is glove mask
    black_mask = ((img_hsv[..., 0] > 0.95) | (img_hsv[..., 0] < 0.05)) & (img_hsv[..., 1] > 0.55)  # empty spaces in the images that are dark and represent the empty space inside the mouse

    # Compute mean values of each channel
    avg_v = np.mean(img_hsv[...,2][~black_mask])#[~np.logical_or(blue_mask, black_mask)])
    avg_r = np.mean(img_float[:, :, 0][~np.logical_or(blue_mask, black_mask)])#[~red_mask])  # Avoid red region
    avg_g = np.mean(img_float[:, :, 1][~np.logical_or(blue_mask, black_mask)])  # Keep green as reference
    avg_b = np.mean(img_float[:, :, 2][~np.logical_or(blue_mask, black_mask)])#[~teeth_mask])  # Avoid white (teeth) region

    # Compute global gray mean
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    factor_scaler = 2

    factor = 0.8 + factor_scaler*((avg_v-1)**2)
    # print(filename)
    # print(avg_v)
    # print(factor)
    # Scale each channel (avoid correcting teeth & red too much)
    img_float[:, :, 0] *= (factor+0.2)*(avg_gray / avg_r)  # Red correction (skip red_mask)
    img_float[:, :, 1] *= factor*(avg_gray / avg_g)  # Green correction (normal)
    img_float[:, :, 2] *= factor*(avg_gray / avg_b)  # Blue correction (skip teeth_mask)

    # Clip values to [0,1] to avoid artifacts
    img_float = np.clip(img_float, 0, 1)

    return img_as_ubyte(img_float)  # Convert back to uint8



import cv2
import os
import json
def calculate_mean_pixel_from_json(image_dir, json_path):
    """
    Calculates the mean pixel value from images listed in the 'teeth_data' key of a JSON file.

    Args:
        image_dir (str): Path to the directory containing the images.
        json_path (str): Path to the JSON file.

    Returns:
        numpy.ndarray: Mean pixel value (RGB).
    """

    image_files = []
    with open(json_path, 'r') as f:
        data = json.load(f)
        for image_info in data.values():
            if "teeth_data" in image_info:
                for tooth_info in image_info["teeth_data"].values():
                    tooth_image_filename = tooth_info["tooth_image_filename"]
                    image_files.append(os.path.join(image_dir, tooth_image_filename))

    if not image_files: 
        print("No image filenames found in the JSON.")
        return None

    means = []
    for image_file in image_files:
        try:
            img = cv2.imread(image_file)
            if img is not None:
                image = dental_gray_world_white_balance(img)
                means.append(np.mean(image, axis=(0, 1)))
            else:
                print(f"Warning: Could not read image {image_file}")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    if not means:
        print("No valid images were processed.")
        return None

    mean_pixel = np.mean(means, axis=0)
    return mean_pixel