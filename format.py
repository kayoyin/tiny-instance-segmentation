import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

filename = "submission.json"
with open(filename, 'r') as file:
    res = json.load(file)

new = []
for entry in res:
    entry["category_id"] = entry["category_id"][0]
    new.append(entry)

with open(filename, 'w') as file:
    json.dump(new, file)

