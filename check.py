import os

import cv2
from tqdm import tqdm

path = 'data/train/images'
for file in tqdm(os.listdir(path)):
    img = cv2.imread('{}/{}'.format(path, file))
    cv2.imwrite('{}/{}'.format(path, file), img)
