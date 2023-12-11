import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA

# cache is a dictionary of embeddings and data, might consider using LRU cache
cache = {}

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
        response = requestServer.infer_test2(frame)
        print(response) # for testing

        # add to cache
        add_to_cache(embedding, response)

        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # stream_client('video_crazyflie.avi')
    stream_client('VIRAT_S_010003_07_000608_000636.avi')