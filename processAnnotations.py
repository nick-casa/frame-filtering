# Parses .objects.txt file and returns a list of lists 
# of bounding boxes for people in each frame
def parse_objects(file_path):

    person_ID = 1
    frame_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            object_ID, duration, frame, left_x, top_y, width, height, object_type = map(int, columns)

            if object_type == person_ID:
                bbox = (left_x, top_y, left_x + width, top_y + height)

                if frame in frame_dict:
                    frame_dict[frame].append(bbox)
                else:
                    frame_dict[frame] = [bbox]

    frames = sorted(frame_dict.items())
    bounding_boxes = [bboxes for _, bboxes in frames]
    return bounding_boxes

if __name__ == '__main__':
    parse_objects('./videos2/VIRAT_S_000200_00_000100_000171.viratdata.objects.txt')