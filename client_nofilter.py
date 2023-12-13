import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA
import json
from collections import OrderedDict
import re
import pickle

# cache is a dictionary of embeddings and data, might consider using LRU cache
cache = OrderedDict()

'''computes SIFT features and returns descriptors'''
def compute_sift_features(frame):

    # initialize SIFT detector
    sift = cv2.SIFT_create()

    # compute SIFT features
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    return descriptors

'''reduces dimensionality of descriptors and returns embedding'''
def compute_embeddings(descriptors):

    # reduce dimensionality by PCA
    pca = PCA(n_components=20) # change this in test depending on data
    embedding = pca.fit_transform(descriptors)
    return embedding.mean(axis=0)

'''adds embedding and data (class label, bounding box, etc.) to cache)'''
def add_to_cache(embedding, data):

    # cache is dictionary of embeddings and data
    key = tuple(embedding) # could consider implementing hash
    cache[key] = data # class label, bounding box, etc.

'''stream client'''
def stream_client(src):

    # load the video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        exit(-1)

    # initialize previous frame
    ret, previous_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        exit(-1)

    frame_no = 0

    # loop through video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(frame_no)

        # compute SIFT features and embeddings
        descriptors = compute_sift_features(previous_frame)
        embedding = compute_embeddings(descriptors)

        # send to server without comparing to cache
        response = requestServer.infer_test2(frame, url="http://20.241.201.181:8080/predictions/fastrcnn")

        matches = re.findall(r'"(?:car|truck)": \[([^\]]*)\],\n    "score": ([0-9.]+)',response)
        # matches = re.findall(r'"person": \[([^\]]*)\],\n    "score": ([0-9.]+)', response)
        boxes = []
        scores = []
        for match in matches:
            box_str, score_str = match
            box_str_cleaned = box_str.replace("\n", "").replace(" ", "")
            box = [int(round(float(item))) for item in box_str_cleaned.split(',')]
            boxes.append(box)

            score = float(score_str)
            scores.append(score)
        
        # add to results
        add_to_cache(embedding, {'bounding_boxes': boxes, 'scores': scores})

        # show bounding boxes
        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # show frame
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()
    with open('client_nofilter_cartest3.pkl', 'wb') as file:
        pickle.dump(cache, file)

if __name__ == '__main__':
    stream_client('./videos2/trimmedVIRAT_S_010113_07_000965_001013.mp4')