import client_LRU
import client_nofilter

def process_video(video):
    # print('start no filter:', video)
    # client_nofilter.stream_client(video)
    # print('end no filter:', video)

    print('start LRU:', video)
    client_LRU.stream_client(video)
    print('end LRU:', video)

if __name__ == "__main__":
    for video in ['./videos2/trimmed_VIRAT_S_010111_09_000981_001014.mp4',
                  './videos2/trimmed_VIRAT_S_010113_07_000965_001013.mp4',
                  './videos2/trimmed_VIRAT_S_050300_04_001057_001122.mp4',
                  './videos2/trimmed_VIRAT_S_050300_07_001623_001690.mp4',
                  './videos2/trimmed_VIRAT_S_050301_03_000933_001046.mp4']:
        process_video(video)