import pickle

def read(pickle_file):
    with open(pickle_file, 'rb') as file:
        pickle_file_loaded = pickle.load(file)

    print(pickle_file_loaded)

if __name__ == '__main__':
    read('/Users/wsethapun/frame-filtering/client_nofilter_short_trim_VIRAT_S_050301_03_000933_001046.pkl')