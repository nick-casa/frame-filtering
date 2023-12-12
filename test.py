import pickle

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
    union_area = float(boxA_area + boxB_area - inter_area)
    iou = inter_area/union_area
    return iou, inter_area, union_area

def find_closest_box(box, boxes):
    highest_iou = 0
    highest_inter = 0
    highest_union = 0
    closest_box = None
    
    # iterate through boxes
    for b in boxes:
        iou,inter,union = bounding_box_accuracy(box, b)
        if iou > highest_iou:
            highest_iou = iou
            highest_inter = inter
            highest_union = union
            closest_box = b
    return closest_box, highest_iou, highest_inter, highest_union

def accuracy_of_bounding():
    with open('client_cache.pkl', 'rb') as file:
        cache = pickle.load(file)
        print("Client\n")
        print(cache)
        print("------------------\n")
    
    with open('ff_client_cache.pkl', 'rb') as file:
        ff_cache = pickle.load(file)
        print("Stream Client\n")
        print(ff_cache)
        print("------------------\n")

    average_frame_acc = []

    # loop over all frames
    for i in range(len(cache)):
        cache_key = list(cache.keys())[i]
        ff_cache_key = list(ff_cache.keys())[i]
        iou_avg = 0
        inter_avg = 0
        union_avg = 0
        count = 0

        # for one frame 
        for box in cache[cache_key]['bounding_boxes']:
            closest_box, iou, inter, union = find_closest_box(box, ff_cache[ff_cache_key]['bounding_boxes'])
            inter_avg += inter
            union_avg += union
        
        average_frame_acc.append(inter_avg/union_avg)

    return average_frame_acc

if __name__ == '__main__':
    average_accuracy = accuracy_of_bounding()
    print("Average IoU accuracy per frame:", average_accuracy)