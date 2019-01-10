
import colorsys
import cv2
import numpy as np


def color_specification(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image."""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1,
                                  image[:, :, n] *
                                  (1 - alpha) + alpha * c,
                                  image[:, :, n])
    return image

def display_instances(image, boxes, masks, ids, names, scores, title = " ",
                            show_mask=True, show_bbox= True, show_pbox=True):
    n_instances = boxes.shape[0]    # Number of objects in the frame
    if not n_instances:
        print('*** Nothing to display ***')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = color_specification(n_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask,color)
        image = cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        image = cv2.putText(image, caption, (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    return image


if __name__ == '__main__':
    import os
    import sys
    import random
    import math
    import time
    import coco
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize_multi

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
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/bauelement20190107T1028/mask_rcnn_bauelement_0029.h5")
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

    # for streaming from Webcams
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)

        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'],title = "Bauelement Detection",
                            show_mask=True, show_bbox= True, show_pbox=True)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

