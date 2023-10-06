import pickle
import pandas as pd
import sys
from pathlib import Path
# path_root = Path('/home/arpit/test_projects/ekhand/')
# sys.path.append(str(path_root))
from src.raw_detections.io import load_detections
# from types import FrameDetections

# obj = pd.read_pickle('assets/EPIC-KITCHENS/P37_101.pkl')
# print(obj._metadata)

# with open('assets/EPIC-KITCHENS/P37_101.pkl', 'rb') as f:
#     x = pickle.load(f)
#     print(''.join(x).decode('utf-8'))


# with open('/home/arpit/EPIC-KITCHENS/P01/hand-objects/P01_03.pkl', "rb") as f:
#     [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]