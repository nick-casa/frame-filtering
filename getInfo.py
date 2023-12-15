import pickle
import test_groundTruth_pr

def printInfo(video):

    file_name_with_extension = video.split('/')[-1]
    file_name = file_name_with_extension.split('.')[0]
    if file_name.startswith('trimmed_'):
        file_name = file_name[len('trimmed_'):]

    annotations = f'./videos2/{file_name}.viratdata.objects.txt'

    pickle_nofilter_results = f'client_nofilter_{file_name}.pkl'
    pickle_nofilter_info = f'client_nofilter_{file_name}_info.pkl'

    pickle_LRU_results = f'client_LRU_{file_name}.pkl'
    pickle_LRU_info = f'client_LRU_{file_name}_info.pkl'
    
    _, nf, lru =test_groundTruth_pr.accuracy_of_bounding(pickle_nofilter_results, pickle_LRU_results, annotations)

    print("\n", file_name, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n")
    print(" \n no filter ------------------ \n")
    
    with open(pickle_nofilter_info, 'rb') as file:
        nofilter_info = pickle.load(file)
    print(nofilter_info)
    print("iou, precision, recall, F1, mAP: ", nf)

    print(" \n LRU ------------------ \n")

    with open(pickle_LRU_info, 'rb') as file:
        lru_info = pickle.load(file)
    print(lru_info)
    print("iou, precision, recall, F1, mAP: ", lru)
    
if __name__ == "__main__":
    for video in ['./videos2/trimmed_VIRAT_S_010111_09_000981_001014.mp4',
                  './videos2/trimmed_VIRAT_S_010113_07_000965_001013.mp4',
                  './videos2/trimmed_VIRAT_S_050300_04_001057_001122.mp4',
                  './videos2/trimmed_VIRAT_S_050300_07_001623_001690.mp4',
                  './videos2/trimmed_VIRAT_S_050301_03_000933_001046.mp4']:
        printInfo(video)