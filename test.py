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

def accuracy_of_bounding():
    with open ('client_cache.json', 'r') as fp:
        cache = json.load(fp)
    
    with open ('ff_client_cache.json', 'r') as fp:
        ff_cache = json.load(fp)

    # cache and ff_cache are ordered dictionaries, compare each entry
    cache_keys = list(cache.keys())
    ff_cache_keys = list(ff_cache.keys())

    # compare each entry
    for i in range(len(cache_keys)):
        cache_key = cache_keys[i]
        ff_cache_key = ff_cache_keys[i]

        # compare bounding boxes
        cache_box = cache[cache_key]['bounding_box']
        ff_cache_box = ff_cache[ff_cache_key]['bounding_box']
        print(bounding_box_accuracy(cache_box, ff_cache_box))

if __name__ == '__main__':
    accuracy_of_bounding()