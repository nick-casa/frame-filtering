import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA
from Caches import LRUCache
import json
from collections import OrderedDict
import re
import pickle

from skimage.metrics import structural_similarity as compare_ssim

test_cache = OrderedDict()

def compute_embeddings(descriptors):
    # reduce dimensionality by PCA
    pca = PCA(n_components=20) # change this in test depending on data
    embedding = pca.fit_transform(descriptors)
    return embedding.mean(axis=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_in_cache(embedding, gray_frame, cache, threshold=0.8):
    highest_ssim = 0
    most_similar_key = None

    keys_list = list(cache.cache.keys())

    for cached_embedding in keys_list:
        cached_gray_frame = cache.get(cached_embedding)['gray_frame']
        score, diff = compare_ssim(gray_frame, cached_gray_frame, full=True)
        if score > highest_ssim:
            highest_ssim = score
            most_similar_key = cached_embedding

    if highest_ssim >= threshold:
        return cache.get(most_similar_key)['response']
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

def compute_avg_distance_between_keypoints(old_frame, current_frame):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

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
    cache = LRUCache(capacity=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = []

        # compute SIFT features and embeddings
        if previous_frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(gray_previous_frame, gray_frame, full=True)

            descriptors = compute_sift_features(frame)
            embedding = compute_embeddings(descriptors)
            cached_response = find_in_cache(embedding, gray_frame, cache, threshold=0.95)

            if cached_response:
                # use cached response
                response = cached_response
                cv2.putText(frame, 'Cached Response Used', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                # perform inference and update cache
                response = requestServer.infer_test2(frame, url="http://20.241.201.181:8080/predictions/fastrcnn")
                cache.put(tuple(embedding), {'response': response, 'gray_frame': gray_frame})

            matches = re.findall(r'"(?:car|truck)": \[([^\]]*)\],\n    "score": ([0-9.]+)',response)
            for match in matches:
                box_str, score_str = match
                box_str_cleaned = box_str.replace("\n", "").replace(" ", "")
                box = [int(round(float(item))) for item in box_str_cleaned.split(',')]
                boxes.append(box)

            add_to_cache(tuple(embedding), {'bounding_boxes': boxes})

            for box in boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

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
    stream_client('VIRAT_S_050300_04_001057_001122.mp4')