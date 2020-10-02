import numpy as np
from json import JSONEncoder
import random
import os
import cv2
from cffi.backend_ctypes import xrange
from pixellib.mask_rcnn import MaskRCNN
from pixellib.config import Config



class configuration(Config):
    NAME = "configuration"


coco_config = configuration(BACKBONE="resnet101", NUM_CLASSES=81, class_names=["BG"], IMAGES_PER_GPU=1,
                            IMAGE_MAX_DIM=1024, IMAGE_MIN_DIM=800, IMAGE_RESIZE_MODE="square", GPU_COUNT=1)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class instance_segmentation():
    def __init__(self):
        self.model_dir = os.getcwd()

    def load_model(self, model_path):
        self.model = MaskRCNN(mode="inference", model_dir=self.model_dir, config=coco_config)
        self.model.load_weights(model_path, by_name=True)

    def segmentImage(self, image_path, output_image_name=None, verbose=None, preferred_classes=[]):

        image = cv2.imread(image_path)
        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_img])

        coco_config.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                                   'bus', 'train', 'truck', 'boat', 'traffic light',
                                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                   'teddy bear', 'hair drier', 'toothbrush']
        r = results[0]

        # apply segmentation mask
        image, box_coordinates, class_labels = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                                 coco_config.class_names, preferred_classes)

        if output_image_name is not None:
            cv2.imwrite(output_image_name, image)
            print("Processed image saved successfully in your current working directory.")

            return image, box_coordinates, class_labels


def display_instances(image, boxes, masks, class_ids, class_name, preferred_classes):
    # determine whether there are more than 3 instances. if that's the case, set n_instances to 3
    if boxes.shape[0] < 3:
        n_instances = boxes.shape[0]
    else:
        n_instances = 3

    # -------- get three masks and consider pref classes -----------
    masks_list, indices = get_masks_of_preferred_class(masks, class_name, class_ids, preferred_classes, n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        colors = random_color()

    box_coordinates = [[] for _ in xrange(n_instances)]
    class_labels = []

    for i in range(n_instances):
        draw_contour_of_mask(masks_list[i], image, colors[i])

        box_coordinates[i] = boxes[indices[i]]

        class_labels.append(class_name[class_ids[indices[i]]])

    return image, np.asarray(box_coordinates), class_labels


def draw_contour_of_mask(mask, image, color):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)


def random_color():
    r = (255, random.randrange(50, 255), random.randrange(50, 255))
    g = (random.randrange(50, 255), 255, random.randrange(50, 255))
    b = (random.randrange(50, 255), random.randrange(50, 255), 255)
    rgb = [r, g, b]
    return rgb


def get_masks_of_preferred_class(masks, classes, class_ids, preferred_classes, n_instances):
    indices_preferred_classes = []
    indices_preferred_masks = []

    print("recognized classes : ", class_ids)

    for i in range(len(preferred_classes)):
        try:
            indices_preferred_classes.append(classes.index(preferred_classes[i]))
        except ValueError:
            print("Class ", preferred_classes[i], "couldn't be found in class array.")



    # --- for every preferred class ---
    for j in range(len(indices_preferred_classes)):
        # --- set found_index true (init) ---
        last_found_index = 0
        start_search_index = 0

        # --- as long as no index was found ---
        while len(indices_preferred_masks) <= n_instances:
            # --- try to find a mask of preferred class ---
            if last_found_index == 0 and start_search_index == 0:
                start_search_index = 0
            elif last_found_index == len(class_ids) - 1:
                break
            else:
                start_search_index = last_found_index + 1

            try:
                last_found_index = class_ids.tolist().index(indices_preferred_classes[j], start_search_index)
                indices_preferred_masks.append(last_found_index)
                start_search_index += 1

            except ValueError:
                print("No class with id ", indices_preferred_classes[j], " found.")
                break;
            else:
                print("Class with id ", indices_preferred_classes[j], " found.")

    # --- check if there are less than max. n_instances. if so, fill list with random masks ---

    if len(indices_preferred_masks) <= n_instances:
        # --- for every missing mask index ---
        for k in range(n_instances - len(indices_preferred_masks)):
            search_for_index = True

            # --- search for an index that is not already in list ---
            while search_for_index:
                # --- get random index ---
                random_mask_index = random.randrange(len(class_ids))

                # --- check if random index is already in mask list. if so, continue search ---
                try:
                    _ = indices_preferred_masks.index(random_mask_index)

                except ValueError:
                    # --- add random mask to list ---
                    indices_preferred_masks.append(random_mask_index)
                    search_for_index = False

    masks_list = []
    for l in range(len(indices_preferred_masks)):
        masks_list.append(masks[:, :, indices_preferred_masks[l]])


    return masks_list, indices_preferred_masks