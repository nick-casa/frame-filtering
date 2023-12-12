import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA
from Caches import LRUCache
import json
from collections import OrderedDict
import re
import pickle

test_cache = OrderedDict()

def compute_embeddings(descriptors):

    # reduce dimensionality by PCA
    pca = PCA(n_components=20) # change this in test depending on data
    embedding = pca.fit_transform(descriptors)
    return embedding.mean(axis=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

'''computes SIFT features and returns descriptors'''
def compute_sift_features(frame):

    # initialize SIFT detector
    sift = cv2.SIFT_create()

    # compute SIFT features
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    return descriptors

'''adds embedding and data (class label, bounding box, etc.) to cache)'''
def add_to_cache(embedding, data):

    # cache is dictionary of embeddings and data
    key = tuple(embedding) # could consider implementing hash
    test_cache[key] = data # class label, bounding box, etc.

def compute_sift_features_count(old_frame, current_frame):
    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Compute SIFT features for both frames
    keypoints_1, descriptors_1 = sift.detectAndCompute(old_frame, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(current_frame, None)

    # BFMatcher or FLANN based matcher can be used
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test and calculate distances
    good_matches = []
    total_distance = 0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            total_distance += m.distance

    # Calculate average distance of good matches
    if len(good_matches) > 0:
        average_distance = total_distance / len(good_matches)
    else:
        average_distance = 0

    return average_distance

def stream_client(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        return

    previous_frame = None
    frame_no = 1
    avg_keypoint_match_distance_sum = 0
    cache = LRUCache(capacity=100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # compute SIFT features and embeddings
        if previous_frame is not None:
            avg_distance = compute_sift_features_count(previous_frame, frame)
            avg_keypoint_match_distance_sum += avg_distance

            descriptors = compute_sift_features(frame)
            embedding = compute_embeddings(descriptors)
            cached_response = find_in_cache(embedding, cache)

            if cached_response and (avg_distance >= (avg_keypoint_match_distance_sum/frame_no)):

                # use cached response
                response = cached_response
                matches = re.findall(r'"person": \[([^\]]*)\]', response)
                boxes = []
                for match in matches:
                    match_cleaned = match.replace("\n", "").replace(" ", "")
                    box = [int(round(float(item))) for item in match_cleaned.split(',')]
                    boxes.append(box)
                
                # add to test cache
                add_to_cache(embedding, {'bounding_boxes': boxes})
                cv2.putText(frame, 'Cached Response Used', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            else:

                # perform inference and update cache
                response = requestServer.infer_test2(frame)
                matches = re.findall(r'"person": \[([^\]]*)\]', response)
                boxes = []
                for match in matches:
                    match_cleaned = match.replace("\n", "").replace(" ", "")
                    box = [int(round(float(item))) for item in match_cleaned.split(',')]
                    boxes.append(box)
                cache.put(tuple(embedding), boxes)

                # put response in test cache for accuracy testing
                add_to_cache(embedding, {'bounding_boxes': boxes})

            cv2.putText(frame, f'Avg Dist: {avg_distance}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.putText(frame, 'No Previous Frame', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()
    with open('ff_client_cache.pkl', 'wb') as file:
        pickle.dump(test_cache, file)

if __name__ == '__main__':
    stream_client('video_crazyflie.avi')