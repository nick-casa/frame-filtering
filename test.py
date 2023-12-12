'''finds the intersection over union accuracy of two bounding boxes'''
def bounding_box_accuracy(boxA, boxB):

    # intersection of rectangles
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # area of intersection
    inter_area = (xB - xA) * (yB - yA)

    # area of other rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # intersection over union
    inter_over_union = inter_area / float(boxA_area + boxB_area - inter_area)

    return inter_over_union

def find_closest_box(box, boxes):
    highest_iou = 0
    closest_box = None
        
    # iterate through boxes
    for b in boxes:
        iou = bounding_box_accuracy(box, b)
        if iou > highest_iou:
            highest_iou = iou
            closest_box = b
    return closest_box, highest_iou

def accuracy_of_bounding():
    with open('client_cache.pkl', 'rb') as file:
        cache = pickle.load(file)
    
    with open('ff_client_cache.pkl', 'rb') as file:
        ff_cache = pickle.load(file)

    total_iou = 0
    count = 0

    for key in cache:
        if key in ff_cache:
            used_boxes = set()
            for box in cache[key]['bounding_boxes']:
                closest_box, iou = find_closest_box(box, ff_cache[key]['bounding_boxes'])
                if closest_box and closest_box not in used_boxes:
                    used_boxes.add(closest_box)
                    total_iou += iou
                    count += 1

    return total_iou / count if count > 0 else 0

if __name__ == '__main__':
    average_accuracy = accuracy_of_bounding()
    print("Average IoU accuracy:", average_accuracy)