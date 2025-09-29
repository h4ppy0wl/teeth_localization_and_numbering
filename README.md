# Teeth Localization and Numbering API

The primary contribution of this work is to provide a robust solution for detecting, localizing, and numbering teeth from smartphone-captured images. It is designed to handle real-world variance in image quality, lighting, focus, and camera types, which is a critical first step for any AI-enabled dental diagnostic pipeline. This project simplifies this step by offering a Dockerized FastAPI application.

The service runs a customized Mask R-CNN model capable of performing tooth localization, numbering (classification), and instance segmentation, returning precise polygonal masks for each detected tooth. It is designed for easy deployment, allowing researchers and developers to quickly integrate advanced dental image analysis into their applications.

## Features

- **FastAPI Backend**: A modern, high-performance web framework for building APIs.
- **Mask R-CNN Model**: A powerful and widely-used model for instance segmentation.
- **Customized for Dentistry**: Includes a modified detection layer and dental-specific image preprocessing to improve accuracy on clinical images.
- **Dockerized**: Packaged for simple, cross-platform deployment.
- **Rich Output**: Returns bounding boxes, class names (tooth numbers), scores, and segmentation polygons.
- **Artifact Generation**: Can optionally generate and serve overlay images and segmentation masks.

---

## Background and Citation

This work is the result of a research project at the **University of Oulu**. The model was trained on a dataset from the **Digileap for Oral Health** project, which aims to develop a virtual reception for oral healthcare. This initiative uses AI and machine learning to analyze smartphone-captured images for remote assessment of treatment needs, making this model particularly well-suited for real-world clinical applications.

The implementation is a customized fork of [z-mahmud22/Mask-RCNN_TF2.14.0](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0), which is itself an updated version of the original Matterport Mask R-CNN for modern TensorFlow 2.x environments.

This version includes significant customizations for the dental domain:
*   **Custom Detection Layer**: The detection layer in `mrcnn/model.py` was modified to better handle the specific challenges of tooth detection, such as managing detections per class and across all classes.
*   **Dental Image Preprocessing**: A custom preprocessing step (`dental_gray_world_white_balance`) has been integrated to normalize images, which is critical for models processing smartphone-captured photos taken under varied lighting conditions.
*   **Pre-trained Dental Weights**: The provided model weights are a key contribution of this work. They are the result of training the customized Mask R-CNN architecture on the specialized Digileap for Oral Health dataset. These weights enable the model to perform accurate tooth localization and numbering out-of-the-box, saving other researchers and developers from the costly and time-consuming process of data collection and model training.

### Citation

This project builds upon the foundational work of the Matterport Mask R-CNN. If you use this code in your research, please provide credit to the original authors.

**Original Matterport Implementation:**
```bibtex
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

---

## Prerequisites

Before you begin, ensure you have [Docker](https://www.docker.com/get-started) installed on your system.

---

## Getting Started

Follow these steps to get the API server up and running.

### 1. Pull the Docker Image

The pre-built Docker image is available on Docker Hub. Pull the latest version with the following command:

```bash
# Replace <your-dockerhub-username> with your actual Docker Hub username
docker pull <your-dockerhub-username>/tooth_localization_numbering_image:latest
```

### 2. Download Model Weights

You need the trained model weights to run inference.

1.  Create a local directory to store the weights:
    ```bash
    mkdir weights
    ```
2.  Download the model weights file (`.h5` file) from the project's repository or release page and place it inside the `weights/` directory you just created. For this example, we'll assume the file is named `mask_rcnn_teeth.h5`.

### 3. Run the Docker Container

Run the Docker container, mapping the API port (`8000`) and mounting your local `weights` directory into the container at `/app/weights`.

```bash
# Make sure you are in the same directory where you created the 'weights' folder
docker run -d \
  -p 8000:8000 \
  -v "$(pwd)/weights:/app/weights" \
  --name teeth_api \
  <your-dockerhub-username>/tooth_localization_numbering_image:latest
```

- `-d`: Runs the container in detached mode.
- `-p 8000:8000`: Maps port 8000 on your host to port 8000 in the container.
- `-v "$(pwd)/weights:/app/weights"`: Mounts your local `weights` directory to `/app/weights` inside the container. This allows the API to access the model file.
- `--name teeth_api`: Assigns a convenient name to the container.

You can check if the container is running with `docker ps`.

---

## Usage

Once the container is running, you can interact with the API.

### Using the Python Client (`find_teeth.py`)

The easiest way to test the API is with the provided client script.

1.  **Set the model weights on the server**: The first step is to tell the server which weights file to use. The client script handles this for you.

2.  **Send an image for prediction**: The script sends an image and retrieves the results.

**Example Command:**

```bash
python3 find_teeth.py \
  -i /path/to/your/image.jpg \
  -w ./weights/mask_rcnn_teeth.h5
```

The script will:
1.  Tell the server to load `mask_rcnn_teeth.h5`.
2.  Send `image.jpg` to the `/predict` endpoint.
3.  Print the JSON response containing the detections.
4.  Download the generated overlay image, mask, and JSON results into a `downloads/` directory.

### Using `curl`

You can also call the API directly using a tool like `curl`.

**1. Set the Weights Path**

First, configure the server to use your model weights file.

```bash
curl -X POST "http://localhost:8000/settings" \
     -H "Content-Type: application/json" \
     -d '{"weights_path": "/app/weights/mask_rcnn_teeth.h5"}'
```

**2. Call Prediction**

Send an image file to the `/predict` endpoint.

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "image=@/path/to/your/image.jpg" \
     -F "return_masked_image=true"
```

The server will respond with a JSON object containing the detection results and URLs to the generated images.

---

## API Reference

### `POST /settings`

Configures the inference settings on the server.

**Request Body (JSON):**
```json
{
  "weights_path": "/app/weights/your_model.h5",
  "confidence_threshold": 0.7,
  "max_detections": 30
}
```

### `GET /settings`

Retrieves the current server settings.

### `POST /predict`

Performs inference on an uploaded image.

**Request (multipart/form-data):**
- `image`: The image file to process.
- `return_mask` (optional, `true` or `false`): If true, generates a transparent PNG of the segmentation masks.
- `return_masked_image` (optional, `true` or `false`): If true, generates a JPG with detections overlaid on the original image.

**Response (JSON):**
A JSON object containing `detections`, `images` (with URLs to generated files), and `meta` information.

### `GET /files/{filename}`

Serves files generated during prediction (e.g., overlays, masks). The URLs are provided in the `/predict` response.

---

## Building from Source (Optional)

If you want to build the Docker image yourself, clone the repository and run the following command from the project root:

```bash
docker build -t <your-dockerhub-username>/tooth_localization_numbering_image:latest .
```
