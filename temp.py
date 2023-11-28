import cv2
import numpy as np
import pickle


file_path = '/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m/P01/hand-objects/P01_101.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    print(data)
