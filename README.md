# frame-filtering

- To use frame filtering, use client_LRU.py, the results of inference will be saved to .pkl files, which can be evaluated for accuracy using test_groundTruth_pr.py.
- To see information about runtime and number of cached frames used and total calls for inference, read the relevant .pkl file using readPickle.py.
- To use client without frame filtering, use client_nofiltering.py.
- Use result .pkl files and dataset annotation files (.objects.txt files) to play videos with bounding boxes via playVideosWithBoxes.py.
