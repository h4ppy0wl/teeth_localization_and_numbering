"""

version: 02.09.2026
training the model with filtered data. Hopefully this is the last experiment! :D


Mask R-CNN
Train on the teeth dataset.

Copyright (c) 2024 ArashNedaei
Licensed under the MIT License (see LICENSE for details)
Written by Arash Nedaei

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 teeth.py train --dataset=/path/to/teeth/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 teeth.py train --dataset=/path/to/teeth/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 teeth.py train --dataset=/path/to/teeth/dataset --weights=imagenet

    # Apply color splash to an image
    python3 teeth.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 teeth.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa
import skimage.io as io
import skimage.color


# Root directory of the project
CODE_DIR = os.path.abspath("../")
if os.name == 'nt':
    DATA_ROOT_DIR = os.path.abspath("Y:/Results/Images/dataset/")
elif os.name == 'posix' and 'linux':
    DATA_ROOT_DIR = os.path.join(CODE_DIR, "dataset")
    
MY_TOOLS_DIR = os.path.abspath("../myTools/")
# Import Mask RCNN
sys.path.append(CODE_DIR)  # To find local version of the library
sys.path.append(MY_TOOLS_DIR)  # To find local version of the library
#from json_crowler import json_crowler #custom functions
from myTools import json_crowler
from myTools.digileap_preprocessing import dental_gray_world_white_balance
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(CODE_DIR, "mask_rcnn_coco.h5")
TEETH_PREVIOUS_TRAINED =  os.path.join(CODE_DIR, "logs/teeth20240902T1640/mask_rcnn_teeth_0100.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(CODE_DIR, "logs/teeth_filtered_training_171224")

import tensorflow as tf #2.9.24

############################################################
#  Configurations
############################################################
def get_api_setting(key, default_value):
    settings_file = "/app/app/settings.json"
    try:
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                settings = json.load(f)
                # Ensure we return the correct type by safely falling back to default
                val = settings.get(key)
                return val if val is not None else default_value
    except Exception:
        pass
    return default_value

class TeethConfig(Config):
    """Configuration for training on the teeth  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "teeth"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 89 #number of validation images. because I want to check the same number of images in each validation session. this approach ensures validation loss fluctuations due to validation data variability


    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # Number of classes (including background)
    # teeth_classes = json_crowler.get_teeth_classes_dict(annotations_json_dir = DATA_ROOT_DIR , json_file_name ="filtered_annotations.json")
    teeth_classes = {
            'T11': 1, 'T12': 2, 'T13': 3, 'T14': 4, 'T15': 5, 'T16': 6, 'T17': 7,
            'T21': 8, 'T22': 9, 'T23': 10, 'T24': 11, 'T25': 12, 'T26': 13, 'T27': 14,
            'T31': 15, 'T32': 16, 'T33': 17, 'T34': 18, 'T35': 19, 'T36': 20, 'T37': 21,
            'T41': 22, 'T42': 23, 'T43': 24, 'T44': 25, 'T45': 26, 'T46': 27, 'T47': 28
        }
    NUM_CLASSES = 1 + len(teeth_classes)  # Background + #tooth

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1103#sweeps all training dataset by 1100! #750 #previously 500

    # # Skip detections with < 80% confidence
    # DETECTION_MIN_CONFIDENCE = 0.5 #previously 0.9
    # Dynamically load from settings.json on container startup
    DETECTION_MIN_CONFIDENCE = get_api_setting("confidence_threshold", 0.5)

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (256, 256)  # (height, width) of the mini-mask #previously (56, 56)

    # Max number of final detections
    #Arash: This parameter is originally used to limit both number of instances of class and also overall detected instances.
    # While for my specific case I need to limit the number of detected instances of class to one (only one tooth with a specific number can be present in an image)
    # and the number of overall detections in an image to 35. So I keep this and add another parameter 
    DETECTION_MAX_INSTANCES = 1#100 #Arash
    #Arash: I have created following parameter to use it in refine_detections_graph()
    # Original mrcnn uses DETECTION_MAX_INSTANCES for limiting number of class detections and overall detections.
    # I keep it to limit class detections, and create DETECTION_MAX_INSTANCES_ALL_CLASSES to limit overal detections
    # DETECTION_MAX_INSTANCES_ALL_CLASSES = 30 #Arash
    # Dynamically load from settings.json on container startup
    DETECTION_MAX_INSTANCES_ALL_CLASSES = get_api_setting("max_detections", 30)



    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.1#0.3

    DETECTION_NMS_THRESHOLD_ALL_CLASSES = 0.7#0.3

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512 #256 04.10.24
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
    
    #these values are selected based on hp2
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.008 # Will not be used in case of dynamic learning rate using LEARNING_RATE_LIST
    LEARNING_RATE_LIST = [0.005, 0.002, 0.0006, 0.0003, 0.0001]# 17.12.24 [0.01, 0.001, 0.0001, 0.00001] #2.9.24 number of elements should be one more than LEARNING_RATE_EPOCH_STEPS
    LEARNING_RATE_EPOCH_STEPS = [10, 20, 40, 80]#18.12.24 for training from 101 to 200____[10, 25, 50]#17.12.24 [7, 15, 30]    #2.10.24
    LEARNING_MOMENTUM = 0.75 #0.75

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor #Arash: or taller?!
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.8]

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 3.,
        "rpn_bbox_loss": 3.,
        "mrcnn_class_loss": 3.,
        "mrcnn_bbox_loss": 3.,
        "mrcnn_mask_loss": 1.,
    }


############################################################
#  Dataset
############################################################

class SingleImageDataset(utils.Dataset):

    def __init__(self):
        super().__init__()

    def add_classes(self, class_names, source = "teeth"):
        for i, c in enumerate(class_names, start=1):
            self.add_class(source, i, c) # starts with 1 not 0. 0 is reserved for background
    
    
    def add_image_from_path(self, image_path, image_id=0, source = "teeth"):
        image_path = os.path.abspath(image_path)
        if not os.path.exists(image_path):
            raise Exception("Image path does not exist: {}".format(image_path))
        self.add_image(source, image_id=image_id, path=image_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
            
        # Arash: Preprocessing    
        final_img = dental_gray_world_white_balance(image)
            
        return final_img
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a teeth dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "teeth":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        h, w = io.imread(info['path']).shape[:2]
        masks = np.zeros((h, w, 0), dtype=np.float64)
        class_ids = np.array([], dtype=np.int32)
        # Return mask, and array of class IDs of each instance. a,b,n = maske.shape , n is the number of classes.
        # we need to export a mask with three dimentions. one 2d mask for each class. we return class lables alongside the mask with the same order of the 3rd dimention (layer == mask)
        # Map class names to class IDs.
        #return mask, num_ids
        return masks, class_ids
        #return mask, object_class_ids


    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "teeth":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = myDataset()
    dataset_train.load_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = myDataset()
    dataset_val.load_data(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    # augmentation = iaa.Sometimes(0.7, iaa.OneOf(
    #                                                 [   iaa.AdditiveGaussianNoise(scale=(0, 0.001*255), per_channel=True),
    #                                                     iaa.Affine(rotate=(-10,10)),
    #                                                     iaa.Multiply((0.9, 1.1)),      
    #                                                 ]
    #                                             )
    #                              )
    # augmentation = [   iaa.AdditiveGaussianNoise(scale=(0, 0.001*255), per_channel=True),
    #                                                     iaa.Affine(rotate=(-10,10)),
    #                                                     iaa.Multiply((0.9, 1.1)),      
    #                                                 ]
    #augmentation = [ iaa.Affine(rotate=(-5,5)), None ]
    # augmentation = [ iaa.AdditiveGaussianNoise(scale=(0, 0.001*255), per_channel=True), iaa.Affine(rotate=(-5,5)), None, None]
    augmentation = [ iaa.Multiply((0.7, 1.1)), iaa.Affine(rotate=(-10,10)), None, None, None]

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # model.py:
    #     self.keras_model.fit(
    #     train_generator,
    #     initial_epoch=self.epoch,
    #     epochs=epochs,
    #     steps_per_epoch=self.config.STEPS_PER_EPOCH,
    #     callbacks=callbacks,
    #     validation_data=val_generator,
    #     validation_steps=self.config.VALIDATION_STEPS,
    #     max_queue_size=100,
    #     workers=4,
    #     use_multiprocessing=True,
    # )
    #defining an adaptive learning rate
    # boundaries = config.LEARNING_RATE_EPOCH_STEPS
    # values = config.LEARNING_RATE_LIST
    # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)

    print("Training 5+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                augmentation=augmentation,
                epochs=100,
                layers="5+")#'heads') hp2 was performed by training heads. I want to first check if additional traning is beneficial. also aumentation is added. it wqas not for hp2

    # print("Train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=40,
    #             augmentation=augmentation,
    #             layers='all')



def single_image_detect(model, image_path, config:TeethConfig):
    """Train the model."""
    # Training dataset.
    ds = SingleImageDataset()
    ds.add_classes(class_names = list(config.teeth_classes.keys()), source="teeth")
    ds.add_image_from_path(image_path = image_path, image_id=0, source="teeth")
    ds.prepare()
    img = ds.load_image(0)
    print("Running on {}".format(image_path))
    print("Image shape: ", img.shape)
    print("class names: ", ds.class_names)

    results = model.detect([img], verbose=1)
    r = results[0]
    print("Class IDs: ", r['class_ids'])
    print("Scores: ", r['scores'])
    print("Masks shape: ", r['masks'].shape)
    print("ROIs shape: ", r['rois'].shape)
    print("Number of instances: ", r['rois'].shape[0])

    return r
   
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TeethConfig()
    else:
        class InferenceConfig(TeethConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "teeth_prev":
        #because of a naming descrepancy I commented previous line and hardcoded following waights path, temporary! 04072024
        weights_path = TEETH_PREVIOUS_TRAINED
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
