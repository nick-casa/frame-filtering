import numpy as np
import cv2
import requestServer
from sklearn.decomposition import PCA
import json
from collections import OrderedDict
import re
import pickle
import time

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

'''stream client'''
def stream_client(src):

    start = time.time()

    result = []

    file_name_with_extension = src.split('/')[-1]
    file_name = file_name_with_extension.split('.')[0]
    if file_name.startswith('trimmed_'):
        file_name = file_name[len('trimmed_'):]

    inference_calls = 0
    frame_no = 0

    # load the video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        exit(-1)

    # loop through video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # send to server without comparing to previous results
        response = requestServer.infer_test2(frame, url="http://20.241.201.181:8080/predictions/fastrcnn")
        inference_calls += 1
        result.append(get_result(response))

        frame_no += 1
        print("frame_no: ", frame_no)

    cap.release()
    cv2.destroyAllWindows()

    end = time.time()

    with open(f'tt_client_nofilter_{file_name}.pkl', 'wb') as file:
        pickle.dump(result, file)

    info = {'total frames': frame_no, 'num_inference_calls': inference_calls, 'runtime': end - start}

    with open(f'tt_client_nofilter_{file_name}_info.pkl', 'wb') as file:
        pickle.dump(info, file)

    results = []

if __name__ == '__main__':
    stream_client('./videos2/trimmedVIRAT_S_010113_07_000965_001013.mp4')