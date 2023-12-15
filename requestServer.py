import requests
import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms as T

def infer(image_input, url="http://20.81.126.214:8080/predictions/fastrcnn"):
    pil_image = Image.fromarray(image_input)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    response = requests.post(url, image_data)
    return response.text

if __name__ == "__main__":
    # Model URL
    url = "http://127.0.0.1:8080/predictions/fastrcnn"
    image_path = "test.png"
    image = cv2.imread(image_path)
    print("Response from model:",  infer(image, url))