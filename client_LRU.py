import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA
from Caches import LRUCache
import json
from collections import OrderedDict
import re
import pickle
import time

from skimage.metrics import structural_similarity as compare_ssim

def compute_embeddings(descriptors):
    # reduce dimensionality by PCA
    pca = PCA(n_components=20) # change this in test depending on data
    embedding = pca.fit_transform(descriptors)
    return embedding.mean(axis=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_in_cache(gray_frame, cache, threshold=0.8):
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


def get_result(response):
    matches = re.findall(r'"(?:car|truck)": \[([^\]]*)\],\n    "score": ([0-9.]+)',response)
    boxes = []
    scores = []
    for match in matches:
        box_str, score_str = match
        box_str_cleaned = box_str.replace("\n", "").replace(" ", "")
        box = [int(round(float(item))) for item in box_str_cleaned.split(',')]
        boxes.append(box)

        score = float(score_str)
        scores.append(score)
    return {'bounding_boxes': boxes, 'scores': scores}

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

    start = time.time()

    result = []

    file_name_with_extension = src.split('/')[-1]
    file_name = file_name_with_extension.split('.')[0]
    if file_name.startswith('trimmed_'):
        file_name = file_name[len('trimmed_'):]

    inference_calls = 0
    used_cache = 0
    frame_no = 0

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        return

    previous_frame = None
    cache = LRUCache(capacity=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(frame_no)

        if frame_no == 60:
            break

        if previous_frame is not None:
            # compute SIFT features and embeddings
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(gray_previous_frame, gray_frame, full=True)

            # descriptors = compute_sift_features(frame)
            # embedding = compute_embeddings(descriptors)
            cached_response = find_in_cache(gray_frame, cache, threshold=0.95)

            # if we found a similar enough image in the cache, use it as a result
            if cached_response:
                used_cache += 1
                print("used cache: ",used_cache)
                response = cached_response
                result.append(get_result(response))

                boxes = result[-1]['bounding_boxes']
                print(boxes)
                for box in boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(frame, 'Cached Response Used', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
            else:
                # perform inference and add to results
                response = requestServer.infer(frame, url="http://20.81.126.214:8080/predictions/fastrcnn")
                inference_calls += 1
                cache.put(frame_no, {'response': response, 'gray_frame': gray_frame})
                result.append(get_result(response))
                boxes = result[-1]['bounding_boxes']
                print(boxes)
                for box in boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        else: 
            response = requestServer.infer(frame, url="http://20.81.126.214:8080/predictions/fastrcnn")
            inference_calls += 1
            result.append(get_result(response))
            boxes = result[-1]['bounding_boxes']
            print(boxes)
            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        frame_no += 1
        print("frame_no: ", frame_no)

    cap.release()
    cv2.destroyAllWindows()

    end = time.time()

    with open(f'demo_client_LRU_{file_name}.pkl', 'wb') as file:
        pickle.dump(result, file)

    info = {'total frames': frame_no, 'num_inference_calls': inference_calls, 'used_cached': used_cache, 'runtime': end - start}

    with open(f'demo_client_LRU_{file_name}_info.pkl', 'wb') as file:
        pickle.dump(info, file)

if __name__ == '__main__':
    stream_client('./videos2/trimmed_VIRAT_S_050301_03_000933_001046.mp4')