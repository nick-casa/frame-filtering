import requests
import json

def infer_image(image_data, url="http://127.0.0.1:8080/predictions/fastrcnn"):
    
    response = requests.post(url, data=image_data)
    return response.text

def infer(image_path, url="http://127.0.0.1:8080/predictions/fastrcnn"):
    # Open the image in binary mode
    with open(image_path, 'rb') as file:
        image_data = file.read()

    response = requests.post(url, data=image_data)

    return response.text

if __name__ == "__main__":
    # Model URL
    url = "http://127.0.0.1:8080/predictions/fastrcnn"
    image_path = "/Users/wsethapun/serve/examples/object_detector/persons.jpg"

    print("Response from model:",  infer(image_path, url))



