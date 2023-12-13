# Parses .objects.txt file and returns a list of lists 
# of bounding boxes for people in each frame
def parse_objects(file_path):

    person_ID = 1
    car_ID = 2
    vehicle_ID = 3
    frame_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            object_ID, duration, frame, left_x, top_y, width, height, object_type = map(int, columns)

            if not frame in frame_dict:
                    frame_dict[frame] = []

            if object_type == car_ID or object_type == vehicle_ID:
                bbox = [left_x, top_y, left_x + width, top_y + height]
                frame_dict[frame].append(bbox)

    frames = sorted(frame_dict.items())

    print(frames)
    bounding_boxes = [bboxes for _, bboxes in frames]

    return bounding_boxes

if __name__ == '__main__':
    parse_objects('./videos2/VIRAT_S_000200_01_000226_000268.mp4')