import numpy as np
import cv2
import requestServer

def compute_sift_features(old_frame, current_frame):
    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Compute SIFT features for both frames
    keypoints_1, descriptors_1 = sift.detectAndCompute(old_frame, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(current_frame, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return len(good_matches)

def stream_client(src):
    # Load the video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to load video.")
        exit(-1)

    previous_frame = None
    frame_no = 1
    match_sum = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is not None:
            match_count = compute_sift_features(previous_frame, frame)
            match_sum += match_count
            if match_count <= match_sum/frame_no:
                cv2.putText(frame, f'Matches: {match_count}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                print(requestServer.inferImage(frame))
            else:
                cv2.putText(frame, f'Matches: {match_count}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 255), 2)


        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream_client('VIRAT_S_010005_04_000299_000323.mp4')