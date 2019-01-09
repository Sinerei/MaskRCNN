import os
import sys
import skimage.io

# import random
# import math
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize_multi
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/bauelement/"))  # To find local version
import Bauelement


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/bauelement20190107T1028/mask_rcnn_bauelement_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# CONFIGURATION
class InferenceConfig(Bauelement.BauelementConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# CREATE MODEL AND LOAD WEIGHTS

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# DEFINE CLASSNAMES

# Bauelement Class names
# Index of the class in the list is its ID. For example, to get ID of
# the Kondensator class, use: class_names.index('Kondensator')
class_names = ['BG','Kondensator','Widerstand']


#RUN OBJECT DETECTION

# Load a random or specific image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
filename=os.path.join(IMAGE_DIR,'board05.jpg')
image = skimage.io.imread(os.path.join(filename))
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


# Run detection

results = model.detect([image], verbose=1)
r = results[0]


#NUMBER OF INSTANCES

count = r['class_ids']
i=0
j=0
for k in range(len(count)):
    if count[k] == 1:
        i=i+1
    elif count[k] == 2:
       j=j+1
print('There are',i,class_names[1])
print('There are',j,class_names[2])

# Visualize results

visualize_multi.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'],title = "Bauelement Detection",
                            show_mask=True, show_bbox= True, show_pbox=True)
