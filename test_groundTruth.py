import pickle
import processAnnotations

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

def accuracy_of_bounding(pickle_nofilter, pickle_LRU, annotation_file_path):
    with open(pickle_nofilter, 'rb') as file:
        cache_nofilter = pickle.load(file)
    
    with open(pickle_LRU, 'rb') as file:
        cache_LRU = pickle.load(file)

    groundTruth_boxes = processAnnotations.parse_objects(annotation_file_path)

    average_frame_acc_LRU = []
    average_frame_acc_nofilter = []

    # loop over all frames for no filtering
    for i in range(len(cache_nofilter)):            
        cache_key = list(cache_nofilter.keys())[i]
        inter_avg = 0
        union_avg = 0

        # for one frame 
        for box in cache_nofilter[cache_key]['bounding_boxes']:
            closest_box, iou, inter, union = find_closest_box(box, groundTruth_boxes[i])
            inter_avg += inter
            union_avg += union
        
        average_frame_acc_nofilter.append(inter_avg/union_avg)
    
        # loop over all frames for LRU
    for i in range(len(cache_LRU)):            
        cache_key = list(cache_LRU.keys())[i]
        inter_avg = 0
        union_avg = 0

        # for one frame 
        for box in cache_LRU[cache_key]['bounding_boxes']:
            closest_box, iou, inter, union = find_closest_box(box, groundTruth_boxes[i])
            inter_avg += inter
            union_avg += union
        
        average_frame_acc_LRU.append(inter_avg/union_avg)

    return average_frame_acc_nofilter, average_frame_acc_LRU

if __name__ == '__main__':
    nofilter, lru = accuracy_of_bounding('client_cache.pkl','ff_client_cache.pkl','./videos2/VIRAT_S_010003_07_000608_000636.viratdata.objects.txt')
    print("Average IoU accuracy per frame, no filtering:", nofilter)
    print("Average IoU accuracy per frame, LRU:", lru)