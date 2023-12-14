import numpy as np
import cv2
from collections import OrderedDict
import pickle
import processAnnotations

def stream(src, pickle_nofilter, pickle_LRU, annotation_file_path):

    with open(pickle_nofilter, 'rb') as file:
        cache_nofilter = pickle.load(file)
    
    with open(pickle_LRU, 'rb') as file:
        cache_LRU = pickle.load(file)

    groundTruth_boxes = processAnnotations.parse_objects(annotation_file_path)

    # load the video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        exit(-1)

    # loop through video
    for i, (ret, frame) in enumerate(iter(lambda: cap.read(), (False, None))):
        if not ret:
            break

        # blue
        for box in cache_nofilter[i]['bounding_boxes']:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        
        #green
        for box in cache_LRU[i]['bounding_boxes']:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        #red
        for box in groundTruth_boxes[i]:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        # show frame
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1000000) & 0xFF == ord('q'):
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream('./videos2/trimmed_VIRAT_S_010111_09_000981_001014.mp4','mr_client_LRU_VIRAT_S_010111_09_000981_001014.pkl','mr_client_LRU_VIRAT_S_010111_09_000981_001014.pkl','./videos2/VIRAT_S_010111_09_000981_001014.viratdata.objects.txt')