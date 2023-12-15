# frame-filtering

- This code makes inference calls to torchserve models, to use a particular torchserve model, change arguments in `client_LRU.py`, `client_nofiltering.py` and `serverRequest.py` to reflect your server address
- To use frame filtering, use `client_LRU.py`, the results of inference will be saved to .pkl files, which can be evaluated for accuracy using `test_groundTruth_pr.py`
- To use client without frame filtering, use `client_nofiltering.py`
- `run_clients.py` can make processing several videos easier
- To see information about runtime and number of cached frames used and total calls for inference, read the relevant `.pkl` file using `readPickle.py`
- `getInfo.py` can make getting information from multiple videos easier
- Use result `.pkl` files and dataset annotation files (`.objects.txt` files) to play videos with bounding boxes via `playVideosWithBoxes.py`

# Running inference server

Navigate to the desired inference model directory
`/fastrcnn` or `/maskrcnn`

- Build the container with: `docker build --tag IMAGE_NAME`

- Run the container with: `docker run --rm -it -p 8080:8080 -p 8081:8081 --name CONTAINER_NAME IMAGE_NAME`
