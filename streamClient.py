import numpy as np
import cv2
import requestServer

def compute_sift_features(old_frame, current_frame):
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
    # Load the video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        exit(-1)

    previous_frame = None
    frame_no = 1
    avg_keypoint_match_distance_sum = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is not None:
            avg_distance = compute_sift_features(previous_frame, frame)
            avg_keypoint_match_distance_sum += avg_distance
        if (avg_distance >= (avg_keypoint_match_distance_sum/frame_no)):
            cv2.putText(frame, f'Avg Dist: {avg_distance}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            # Perform inference
            ##print(requestServer.inferImage(frame))
        else:
            cv2.putText(frame, f'Avg Dist: {avg_distance}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (1, 1, 255), 2)
            # Cache lookup


        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream_client('video_crazyflie.avi')