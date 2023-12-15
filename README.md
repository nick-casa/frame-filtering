
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

# Frame-filtering

- This code makes inference calls to torchserve models, to use a particular torchserve model, change arguments in `client_LRU.py`, `client_nofiltering.py` and `serverRequest.py` to reflect your server address
- To use frame filtering, use `python client_LRU.py`, the results of inference will be saved to .pkl files, which can be evaluated for accuracy using `test_groundTruth_pr.py`
- To use client without frame filtering, use `python client_nofiltering.py`
- `run_clients.py` can make processing several videos easier
- To see information about runtime and number of cached frames used and total calls for inference, read the relevant `.pkl` file using `readPickle.py`
- `python getInfo.py` can make getting information from multiple videos easier
- Use result `.pkl` files and dataset annotation files (`.objects.txt` files) to play videos with bounding boxes via `python playVideosWithBoxes.py`
