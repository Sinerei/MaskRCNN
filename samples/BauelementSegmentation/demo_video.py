
import cv2
from mrcnn.visualize_video import display_instances
import os
import sys

VIDEO_NAME = 'kondensators.mp4'

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/bauelement/"))  # To find local version
from samples.bauelement import Bauelement

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/bauelement20190107T1028/mask_rcnn_bauelement_0010.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(Bauelement.BauelementConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'Kondensator', 'Widerstand']

# Detection
capture = cv2.VideoCapture(VIDEO_NAME)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('masked_video.avi', codec, 20.0, size)

# i = 0
# frame_rate_divider = 3
while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:

        # Add Mask to frames
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'], title="Bauelement Detection")
        # if i % frame_rate_divider == 0:
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        output.write(frame)
        cv2.imshow('frame', frame)
        #     i += 1
        # else:
        #     i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
