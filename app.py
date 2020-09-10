import base64
import os
from io import BytesIO
from PIL import Image
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
            image = request.json['image']
            image_string = base64.b64decode(image)
            image_data = BytesIO(image_string)
            img = Image.open(image_data)
            img.save(INPUT_IMAGE)

            preferred_classes = request.json['classes']
            instance_segment_image = instance_segmentation()
            instance_segment_image.load_model(INSTANCE_MODEL)
            segmask, output = instance_segment_image.segmentImage(INPUT_IMAGE, output_image_name=OUTPUT_IMAGE, show_bboxes=False, preferred_classes = preferred_classes)

            with open(OUTPUT_IMAGE, "rb") as img_file:
                my_string = base64.b64encode(img_file.read())
                final_base64_image_string = my_string.decode('utf-8')
            return {"output_image": final_base64_image_string}

        return "error"


api.add_resource(InstanceSegmentation, '/api')

if __name__ == '__main__':
    app.run(debug=True)