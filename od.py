import json
import requests
import time

from flask import Flask, jsonify, request
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

app = Flask(__name__)

# Initialize the object detection pipeline
object_detector = pipeline("object-detection")

def not_whitespace_string(input_string):
    if isinstance(input_string, str):
        return input_string.strip() == ''
    else:
        raise ValueError("Input is not a string")

OBJECT_FIELD_NAME = "object"

def fetch_profile_image(profile_image):
    response = requests.get(profile_image)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.show()
        return image
    else:
        return None

class ProxyError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

def fetch_so_top_users(page=1, pagesize=10, order='desc', sort='reputation'):
    url = "https://api.stackexchange.com/2.2/users"

    # Send the GET request with parameters to fetch the top users
    params = {
        'site': 'stackoverflow',
        'page': page,
        'pagesize': pagesize,
        'order': order,
        'sort': sort
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise ProxyError(f"Call to StackOverflow failed with status code: {response.status_code}", response.status_code)

    json_data = response.json()
    return json_data.get('items', [])

def detect_object(image, obj_name):
    """
    Detect if there is an object specified in `obj_name`.

    Parameters:
    image: an Pillo Image object
    obj_name: The name of the object to detect, eg.: 'person' 'dog' etc.
    """
    obj_boxes = []
    bounding_boxes = object_detector(image)
    for box in bounding_boxes:
        if box["label"] == obj_name:
            obj_boxes.append(box)
    return obj_boxes

@app.route('/api/v1/users', methods=['POST'])
def detections():
    data = request.get_json()
    print(data)

    if 'query' not in data:
        return jsonify({'error': "Query parameter 'query' is required"}), 400

    query = data['query']

    # Extract a specific field from the JSON data
    if not query or OBJECT_FIELD_NAME not in query:
        return jsonify({"error": "Field 'object' not found in the query"}), 400

    object_name = query[OBJECT_FIELD_NAME]
    if not isinstance(object_name, str):
        return jsonify({"error": "Field 'object' is not a string"}), 400

    object_name = object_name.strip()
    if object_name.strip() == '':
        return jsonify({"error": "Field 'object' is white space"}), 400

    try:
        users = fetch_so_top_users()
    except ProxyError as e:
        return jsonify({"error": e.message}), 500

    detection_results = []
    for user in users:
        profile_image = fetch_profile_image(user['profile_image'])

        user_id = user['user_id']
        display_name = user['display_name']
        profile_image = user['profile_image']

        start_time = time.time()  # Record the start time
        if profile_image is not None:
            detection_boxes = detect_object(profile_image, query["object"])
        end_time = time.time()  # Record the end time
        latency = (end_time - start_time) * 1000

        detection_results.append({
            "user_id": user_id,
            "display_name": display_name,
            "profile_image": profile_image,
            "object_detected": len(detection_boxes) > 0,
            "bounding_boxes": detection_boxes,
            "detection_time_ms": latency,
        })

    return jsonify(detection_results), 201

if __name__ == '__main__':
    app.run(debug=True)
