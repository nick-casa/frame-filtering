import pickle

def read(pickle_file):
    with open(pickle_file, 'rb') as file:
        pickle_file_loaded = pickle.load(file)

    print(pickle_file_loaded)

if __name__ == '__main__':
    read('client_nofilter_trimmedVIRAT_S_010113_07_000965_001013_info.pkl')