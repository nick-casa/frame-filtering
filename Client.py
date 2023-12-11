import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA
import LRUCache

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


''' finds the cosine similarity between a and b (embeddings)'''
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.lingalg.norm(a) * np.lingalg.norm(b))

''' finds the most similar images in cache (above threshold) '''
def find_in_cache(embedding, cache, threshold=0.8):
    most_similar = None
    highest_similarity = 0

    for cached_embedding in cache.cache.keys():
        similarity = cosine_similarity(embedding, np.array(cached_embedding))
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar = cached_embedding

    if highest_similarity >= threshold:
        return cache.get(most_similar)
    else:
        return None


'''stream client'''
def stream_client(src, cache_size=100):

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

    cache = LRUCache(capacity=cache_size)
    frame_no = 0

    # loop through video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #print(frame_no)
        # compute SIFT features and embeddings
        descriptors = compute_sift_features(previous_frame)
        embedding = compute_embeddings(descriptors)

        cached_response = find_in_cache(embedding, cache)
        if cached_response is not None:
            response = cached_response
        else:
            response = requestServer.infer_test2(frame)
            cache.put(tuple(embedding), response)

        print(frame_no, response)
        # send to server without comparing to cache
        #response = requestServer.infer_test2(frame)
        #print(response) # for testing
        # add to cache
        #add_to_cache(embedding, response)

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