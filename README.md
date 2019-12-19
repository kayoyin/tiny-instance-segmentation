# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow by matterport. 
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

Here is the [original implementation](https://github.com/matterport/Mask_RCNN). 

Changes in this repository include:
* Training scheme for finetuning pre-trained model on Imagenet
* Addition of heavier data augmentation
* Minor changes to model configuration



## Reproducing Submission
To reproduce my submission, do the following steps:
1. [Installation](https://github.com/kayoyin/image-segmentation#Installation)
2. [Training](https://github.com/kayoyin/image-segmentation#Training)
3. [Inference](https://github.com/kayoyin/image-segmentation#Inference)

## Installation

It is strongly recommended to set up a new environment for this project. An example of creating a new virtual environment and installing dependencies is given below.
```
virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
``` 

## Training

Below are three useful commands to train your model on the dataset given in this repository. 
Model weights and tensorboard logs will be saved in `logs/`

```
# Train a new model starting from ImageNet weights
python samples/coco/coco.py train --model=imagenet

# Continue training a model that you had trained earlier
python samples/coco/coco.py train --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python samples/coco/coco.py train --model=last
```


The training schedule, learning rate, and other parameters should be set in `samples/coco/coco.py`.

Sometimes, the following error appears:
```
Exception: Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc. 
```

In this case, change `IMAGE_MAX_DIM=1024` in `mrcnn/configs.py`, which will allow the model to run with similar performance and slower training.

## Inference
Inference and formatting the submission file is done as follows.

```
python samples/coco/coco.py evaluate --model=last
python format.py
```

This will save the final submission in the COCO format in `submission.json`

## Differences from the Official Paper
This implementation follows the Mask RCNN paper for the most part, but there are a few cases where we deviated in favor of code simplicity and generalization. These are some of the differences we're aware of. If you encounter other differences, please do let us know.

* **Image Resizing:** To support training multiple images per batch we resize all images to the same size. For example, 1024x1024px on MS COCO. We preserve the aspect ratio, so if an image is not square we pad it with zeros. In the paper the resizing is done such that the smallest side is 800px and the largest is trimmed at 1000px.
* **Bounding Boxes**: Some datasets provide bounding boxes and some provide masks only. To support training on multiple datasets we opted to ignore the bounding boxes that come with the dataset and generate them on the fly instead. We pick the smallest box that encapsulates all the pixels of the mask as the bounding box. This simplifies the implementation and also makes it easy to apply image augmentations that would otherwise be harder to apply to bounding boxes, such as image rotation.

    To validate this approach, we compared our computed bounding boxes to those provided by the COCO dataset.
We found that ~2% of bounding boxes differed by 1px or more, ~0.05% differed by 5px or more, 
and only 0.01% differed by 10px or more.

* **Learning Rate:** The paper uses a learning rate of 0.02, but we found that to be
too high, and often causes the weights to explode, especially when using a small batch
size. It might be related to differences between how Caffe and TensorFlow compute 
gradients (sum vs mean across batches and GPUs). Or, maybe the official model uses gradient
clipping to avoid this issue. We do use gradient clipping, but don't set it too aggressively.
We found that smaller learning rates converge faster anyway so we go with that.
