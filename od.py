import json
from flask import Flask, jsonify, request
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Load font
font = ImageFont.truetype("font.ttf", 40)

# Initialize the object detection pipeline
object_detector = pipeline("object-detection")

with Image.open("image.png") as im:
    # Perform object detection
    bounding_boxes = object_detector(im)
    print(bounding_boxes)

def not_whitespace_string(input_string):
    if isinstance(input_string, str):
        return input_string.strip() == ''
    else:
        raise ValueError("Input is not a string")

OBJECT_FIELD_NAME = "object"

@app.route('/detections', methods=['POST'])
def detections():
    # Dummy implementation for POST request
    data = request.get_json()
    print(data)

    # Extract a specific field from the JSON data
    if not data or OBJECT_FIELD_NAME not in data:
        return jsonify({"error": "Field 'object' not found in the JSON data"}), 400

    field_value = data[OBJECT_FIELD_NAME]
    if not isinstance(field_value, str):
        return jsonify({"error": "Field 'object' is not a string"}), 400

    field_value = field_value.strip()
    if field_value.strip() == '':
        return jsonify({"error": "Field 'object' is white space"}), 400

    # Open the image
    with Image.open("image.png") as im:
        # Perform object detection
        bounding_boxes = object_detector(im)
        for box in bounding_boxes:
            if box["label"] == field_value:
                return jsonify(box), 201
        return jsonify(bounding_boxes), 404

if __name__ == '__main__':
    app.run(debug=True)

