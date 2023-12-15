import client_LRU
import client_nofilter

def process_video(video):
    print('start no filter:', video)
    client_nofilter.stream_client(video)
    print('end no filter:', video)

    print('start LRU:', video)
    client_LRU.stream_client(video)
    print('end LRU:', video)

if __name__ == "__main__":
    for video in ['./videos2/trimmed_VIRAT_S_050301_03_000933_001046.mp4']:
        process_video(video)