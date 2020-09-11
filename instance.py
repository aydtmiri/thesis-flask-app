import cv2
import numpy as np
import random
import os
import cv2
from pixellib.mask_rcnn import MaskRCNN
from pixellib.config import Config
import colorsys


from skimage import measure


class configuration(Config):
    NAME = "configuration"


coco_config = configuration(BACKBONE="resnet101", NUM_CLASSES=81, class_names=["BG"], IMAGES_PER_GPU=1,
                            IMAGE_MAX_DIM=1024, IMAGE_MIN_DIM=800, IMAGE_RESIZE_MODE="square", GPU_COUNT=1)


class instance_segmentation():
    def __init__(self):
        self.model_dir = os.getcwd()

    def load_model(self, model_path):
        self.model = MaskRCNN(mode="inference", model_dir=self.model_dir, config=coco_config)
        self.model.load_weights(model_path, by_name=True)

    def segmentImage(self, image_path, show_bboxes=False, output_image_name=None, verbose=None, preferred_classes = []):

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
        if show_bboxes == False:

            # apply segmentation mask
            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, preferred_classes)
            #
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            # apply segmentation mask with bounding boxes
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names,
                                           r['scores'])

            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed Image saved successfully in your current working directory.")
            return r, output


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5): #alpha = opacity

    for c in range(3):

        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_name,preferred_classes):

    get_masks_of_preferred_class(masks,class_name,preferred_classes)

    # detmerine wheteher there are more than 3 instances. if that's the case, set n_instances to 3
    if boxes.shape[0] < 3:
        n_instances = boxes.shape[0]
    else:
        n_instances = 3

    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        mask = masks[:, :, i]
        print(class_name[class_ids[i]]) #Classnames of mask

        image = apply_mask(image, mask, color)

    return image


def get_masks_of_preferred_class(masks,classes, preferred_classes):

    index = classes.index(preferred_classes)
    print("index", index)


def display_box_instances(image, boxes, masks, class_ids, class_name, scores):

    # detmerine wheteher there are more than 3 instances. if that's the case, set n_instances to 3
    if boxes.shape[0] < 3:
        n_instances = boxes.shape[0]
    else:
        n_instances = 3
    colors = random_colors(n_instances)

    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_name[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        color_rec = [int(c) for c in np.array(colors[i]) * 255]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rec, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(255, 255, 255))

    return image


