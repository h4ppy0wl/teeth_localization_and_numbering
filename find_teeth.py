import requests
from urllib.parse import urljoin
import os
import sys

import argparse
BASE_URL = "http://localhost:8000"  # or your server URL

def set_server_settings(weights_path: str, confidence: float = 0.7):
    """Set the default settings on the server."""
    settings_url = urljoin(BASE_URL, "/settings")
    payload = {
        "weights_path": weights_path,
        "confidence_threshold": confidence
    }
    print(f"Updating server settings: {payload}")
    r = requests.post(settings_url, json=payload, timeout=10)
    r.raise_for_status()
    print("Server settings updated successfully.")
    return r.json()


def call_predict(image_path: str, return_mask: bool = True, return_masked: bool = True):
    """Call the predict endpoint with an image."""
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
        data = {
            # booleans must be strings in multipart forms
            "return_mask": "true" if return_mask else "false",
            "return_masked_image": "true" if return_masked else "false",
            # optional:
            # "confidence_threshold": "0.6",
            # "max_detections": "20",
        }
        r = requests.post(urljoin(BASE_URL, "/predict"), files=files, data=data, timeout=120)
        r.raise_for_status()
        return r.json()

def download_file(url_path, dest_dir="downloads"):
    """
    url_path can be relative (e.g., /files/xyz.png) or absolute.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = urljoin(BASE_URL, url_path)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        # choose filename from URL
        filename = os.path.basename(r.url.split("?")[0])
        out_path = os.path.join(dest_dir, filename)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Client for the Mask R-CNN Teeth Detection API.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image file.")
    parser.add_argument("-w", "--weights", required=False, help="Optional: Path to the model weights file on the host.")
    args = parser.parse_args()

    # Validate that the input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        sys.exit(1)

    try:
        # If weights are provided, set them on the server.
        if args.weights:
            # The API needs the path as it exists *inside* the container.
            # We assume the host's weights directory is mounted to /app/weights.
            weights_filename = os.path.basename(args.weights)
            container_weights_path = f"/app/weights/{weights_filename}"

            # Set the weights path on the server.
            set_server_settings(weights_path=container_weights_path)

        # Call prediction with the user's image.
        print(f"\nRequesting prediction for '{args.image}'...")
        resp = call_predict(args.image, return_mask=True, return_masked=True)
        print("\n--- Prediction Result ---")
        print("Detections:", resp["detections"])
        
        # Download any files the API produced
        for key in ("mask_url", "masked_url", "json_url"):
            if resp["images"].get(key):
                saved_path = download_file(resp["images"][key])
                print(f"Saved {key} to -> {saved_path}")
        print("-----------------------\n")
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")
        print("Is the Docker container running and the port mapped correctly?")
