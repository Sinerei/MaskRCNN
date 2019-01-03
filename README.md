# MaskRCNN
Implementation of MaskRCNN for detecting Bauelemente

Contributing

Contributions to this repository are welcome. Examples of things you can contribute:

    Speed Improvements. Like re-writing some Python code in TensorFlow or Cython.
    Training on other datasets.
    Accuracy Improvements.
    Visualizations and examples.


Requirements

Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in requirements.txt.
MS COCO Requirements:

To train or test on MS COCO, you'll also need:

    pycocotools (installation instructions below)
    MS COCO Dataset
    Download the 5K minival and the 35K validation-minus-minival subsets. More details in the original Faster R-CNN implementation.

If you use Docker, the code has been verified to work on this Docker container.


Installation

    Install dependencies

    pip3 install -r requirements.txt

    Clone this repository

    Run setup from the repository root directory

    python3 setup.py install

    Download pre-trained COCO weights (mask_rcnn_coco.h5) from the releases page.

    (Optional) To train or test on MS COCO install pycocotools from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).
        Linux: https://github.com/waleedka/coco
        Windows: https://github.com/philferriere/cocoapi. You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
