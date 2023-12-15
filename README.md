# Requirements

1. Create a python venv to run code in (the code was developed with Python 3.8):

```
python3.8 -m venv .venv
```

2. Activate the venv (the venv should be active any time you are running code from the repo).

```
source .venv/bin/activate
```

3. Install requirements

```
pip install torch
pip install torchvision
pip install numpy
pip install scikit-learn
pip install opencv-python
pip install mapcalc
pip install jsonlines
```

# Running inference server

Navigate to the desired inference model directory
`/fastrcnn` or `/maskrcnn`

- Build the container with: `docker build --tag IMAGE_NAME`

- Run the container with: `docker run --rm -it -p 8080:8080 -p 8081:8081 --name CONTAINER_NAME IMAGE_NAME`

# Building Image on Azure

1. Install the Azure CLI on your local machine and log in to your Azure account using `az login`.
2. Create an ACR instance to store your Docker images.
   `az acr create --resource-group myResourceGroup --name myRegistry --sku Basic`
3. Use the Azure CLI to fetch login credentials for your ACR and configure Docker to use them.
   `az acr login --name myRegistry`
4. Tag your Docker image with the login server name of your ACR.
   `docker tag IMAGE_NAME myRegistry.azurecr.io/IMAGE_NAME:v1`
5. Upload your Docker image to the ACR.
   `docker push myRegistry.azurecr.io/IMAGE_NAME:v1`
6. Create an ACI instance with your Docker image from ACR.
   `az container create --resource-group myResourceGroup --name myContainer --image myRegistry.azurecr.io/IMAGE_NAME:v1 --cpu 1 --memory 1.5 --registry-login-server myRegistry.azurecr.io --registry-username <acr-username> --registry-password <acr-password> --dns-name-label myLabel --ports 8080 8081`
7. Check the status and logs of your container instance.
   `az container logs --resource-group myResourceGroup --name myContainer`

Replace placeholders (myResourceGroup, myRegistry, IMAGE_NAME, etc.) with actual resource names and credentials.

# Frame-filtering

- This code makes inference calls to torchserve models, to use a particular torchserve model, change arguments in `client_LRU.py`, `client_nofiltering.py` and `serverRequest.py` to reflect your server address
- To use frame filtering, add the file path to the main function of `client_LRU.py` and run `python client_LRU.py`. The results of inference will be saved to `.pkl` files. We've provided a sample video `trimmed_VIRAT_S_050301_03_000933_001046.mp4`
- Evaluate for accuracy by modifying file paths of `.pkl` files and video annotation files in the main function of. `test_groundTruth_pr.py`. We've the annotations for the sample video `VIRAT_S_050301_03_000933_001046.viratdata.objects.txt`
- To use client without frame filtering run `python client_nofiltering.py`
- `run_clients.py` can make processing several videos easier
- To see information about runtime and number of cached frames used and total calls for inference, read the relevant `.pkl` file using `readPickle.py`
- `python getInfo.py` can make getting information from multiple videos easier
- Use result `.pkl` files and dataset annotation files (`.objects.txt` files) to play videos with bounding boxes via `python playVideosWithBoxes.py`
