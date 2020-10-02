import base64
import json
import os
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource

from instance import instance_segmentation

INSTANCE_MODEL = "mask_rcnn_coco.h5"

INPUT_IMAGE = "images/input.jpg"
OUTPUT_IMAGE = "output_images/output.jpg"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
api = Api(app)


class InstanceSegmentation(Resource):
    def post(self):
        if request.json:
            post_request = request.json['data'][0]

            # get base64 string of image from json request
            image = post_request['image']

            # decode base64 to image
            image_string = base64.b64decode(image)
            image_data = BytesIO(image_string)
            img = Image.open(image_data)

            # save image
            img.save(INPUT_IMAGE)

            preferred_classes = request.json['data'][0]['classes']
            instance_segment_image = instance_segmentation()
            instance_segment_image.load_model(INSTANCE_MODEL)

            try:
                image, box_coordinates, class_labels = instance_segment_image.segmentImage(INPUT_IMAGE,
                                                                                           output_image_name=OUTPUT_IMAGE,
                                                                                           preferred_classes=preferred_classes)
            except Exception as e:
                return {"output_image": "", "box_coordinates": "",
                        "class_labels": "", "error": format(e)}

            with open(OUTPUT_IMAGE, "rb") as img_file:
                # convert processed image to base64
                my_string = base64.b64encode(img_file.read())
                final_base64_image_string = my_string.decode('utf-8')

            boxes_json = json.dumps({"output_image": final_base64_image_string, "box_coordinates": box_coordinates,
                                     "class_labels": class_labels}, cls=NumpyEncoder)
            return {"output_image": final_base64_image_string, "box_coordinates": box_coordinates.tolist(),
                    "class_labels": class_labels, "error": ""}

        return "error"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


api.add_resource(InstanceSegmentation, '/api')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
