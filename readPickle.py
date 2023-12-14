import pickle

def read(pickle_file):
    with open(pickle_file, 'rb') as file:
        pickle_file_loaded = pickle.load(file)

    print(pickle_file_loaded)

if __name__ == '__main__':
    read('mr_client_LRU_VIRAT_S_010111_09_000981_001014_info.pkl')