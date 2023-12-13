import numpy as np
import pickle
import processAnnotations
from mapcalc import calculate_map, calculate_map_range
import copy

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
    closest_box_idx = 0
        
    # iterate through boxes
    for i,b in enumerate(boxes):
        if b is None:
            continue
        iou,inter,union = bounding_box_accuracy(box, b)
        if iou > highest_iou:
            highest_iou = iou
            highest_inter = inter
            highest_union = union
            closest_box = b
            closest_box_idx - i 
    return closest_box, closest_box_idx, highest_iou, highest_inter, highest_union

def precision_recall(modelBoxes, groundTruthBoxes, threshold = 0.4):
    TP = 0
    FP = 0
    FN = 0
    num_groundTruth = len(groundTruthBoxes)
    groundTruthUnused = np.ones(num_groundTruth)
    groundTruthBoxesCache = copy.deepcopy(groundTruthBoxes)
    
    for model_idx, modelBox in enumerate(modelBoxes):
        if model_idx >= num_groundTruth:
            break
        _, ground_idx, iou, _, _, = find_closest_box(modelBox, groundTruthBoxesCache)
        groundTruthBoxesCache[ground_idx] = None
        groundTruthUnused[ground_idx] = 0
        if iou >= threshold:
            TP += 1
        else:
            FP += 1
    
    FN += np.sum(groundTruthUnused)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall

def frame_iou(modelBoxes, groundTruthBoxes):
    inter_avg = 0
    union_avg = 0

    for box in modelBoxes:
        _, _, _, inter, union = find_closest_box(box, groundTruthBoxes)
        inter_avg += inter
        union_avg += union

    if union_avg == 0:
        return 0
    
    return inter_avg/union_avg

def frame_mapcalc(modelBoxes, modelScores, groundTruthBoxes):
    groundTruth = {'boxes': groundTruthBoxes, 'labels' : np.ones(len(groundTruthBoxes))}
    resultDict =  {'boxes':modelBoxes, 'labels' : np.ones(len(modelBoxes)), 'scores': modelScores}

    return calculate_map(groundTruth, resultDict, 0.5)


def accuracy_of_bounding(pickle_nofilter, pickle_LRU, annotation_file_path):
    with open(pickle_nofilter, 'rb') as file:
        cache_nofilter = pickle.load(file)
    
    with open(pickle_LRU, 'rb') as file:
        cache_LRU = pickle.load(file)

    groundTruthBoxes_allFrames = processAnnotations.parse_objects(annotation_file_path)

    iou_acc_LRU = []
    pr_acc_LRU = []
    mAP_LRU = []

    iou_acc_nofilter = []
    pr_acc_nofilter = []
    mAP_nofilter = []

    frame_similarity = []


    # loop over all frames
    for i in range(len(cache_nofilter)):
        cache_nofilter_key = list(cache_nofilter.keys())[i]
        cache_LRU_key = list(cache_LRU.keys())[i]
        nofilterBoxes = cache_nofilter[cache_nofilter_key]['bounding_boxes']
        nofilterScore = cache_nofilter[cache_nofilter_key]['scores']
        lruBoxes = cache_LRU[cache_LRU_key]['bounding_boxes']
        lruScore = cache_nofilter[cache_LRU_key]['scores']
        
        groundTruthBoxes = groundTruthBoxes_allFrames[i]

        frame_similarity.append(frame_iou(nofilterBoxes, lruBoxes))
        iou_acc_nofilter.append(frame_iou(nofilterBoxes, groundTruthBoxes))
        iou_acc_LRU.append(frame_iou(lruBoxes, groundTruthBoxes))

        pr_acc_nofilter.append(precision_recall(nofilterBoxes, groundTruthBoxes_allFrames[i]))
        pr_acc_LRU.append(precision_recall(lruBoxes, groundTruthBoxes_allFrames[i]))

        mAP_nofilter.append(frame_mapcalc(nofilterBoxes, nofilterScore,groundTruthBoxes))
        mAP_LRU.append(frame_mapcalc(lruBoxes, lruScore,groundTruthBoxes))

    return frame_similarity, iou_acc_nofilter, iou_acc_LRU, pr_acc_nofilter, pr_acc_LRU, mAP_nofilter, mAP_LRU

if __name__ == '__main__':
    frame_similarity, iou_acc_nofilter, iou_acc_LRU, pr_acc_nofilter, pr_acc_LRU, mAP_nofilter, mAP_LRU = accuracy_of_bounding('client_nofilter_cartest3.pkl','client_nofilter_cartest3.pkl','./videos2/VIRAT_S_010113_07_000965_001013.viratdata.objects.txt')
    print("Average IoU similarity per frame between no filtering and LRU:", frame_similarity, "\n")
    print("Average IoU accuracy per frame compared to ground truth, no filtering:", iou_acc_nofilter, "\n")
    print("Average IoU accuracy per frame compated to ground truth LRU:", iou_acc_LRU, "\n")
    print("Precision and recall per frame compared to ground truth, no filtering:", pr_acc_nofilter, "\n")
    print("Precision and recall per frame compated to ground truth LRU:", pr_acc_LRU, "\n")
    print("mAP scores, no filtering:", mAP_nofilter, "\n")
    print("mAP scores, LRU:", mAP_LRU, "\n")