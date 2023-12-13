import requests
import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms as T

# adapted from https://github.com/pytorch/serve/issues/760
def preprocess(img_path_or_buf):

    raw_image = (
        Image.fromarray(cv2.imread(img_path_or_buf))
        if isinstance(img_path_or_buf, str)
        else img_path_or_buf
    )
    # If buffer was np.array instead of PIL.Image, transform it
    if type(raw_image) == np.ndarray:
        raw_image = Image.fromarray(raw_image)

    # Converts the image to RGB
    raw_image = raw_image.convert("RGB")

    # Define the transformations
    image_processing = T.Compose([
        # T.Resize(256),
        # T.CenterCrop(224),
        T.ToTensor(),  # Convert to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.ToPILImage()
        ])

    pil_image = image_processing(raw_image)

    raw_image_bytes = BytesIO()
    pil_image.save(raw_image_bytes, format = "PNG")
    raw_image_bytes.seek(0)
    return raw_image_bytes.read()

def infer_image(image_data, url="http://127.0.0.1:8080/predictions/fastrcnn"):
    response = requests.post(url, image_data)
    return response.text

# This matches the example 100%
def infer(image_path, url="http://127.0.0.1:8080/predictions/fastrcnn"):
    # Open the image in binary mode
    with open(image_path, 'rb') as file:
        image_data = file.read()
    response = requests.post(url, image_data)
    return response.text

def infer_test(image, url="http://127.0.0.1:8080/predictions/fastrcnn"):
    image_data = preprocess(image)
    response = requests.post(url, image_data)
    return response.text

def infer_test2(image_input, url="http://127.0.0.1:8080/predictions/fastrcnn"):
    pil_image = Image.fromarray(image_input)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    response = requests.post(url, image_data)
    return response.text

if __name__ == "__main__":
    # Model URL
    url = "http://20.81.126.214:8080/predictions/fastrcnn"
    image_path = "test.png"
    image = cv2.imread(image_path)
    print("Response from model:",  infer_test2(image, url))