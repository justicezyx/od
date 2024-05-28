import json
import requests
import time

from flask import Flask, jsonify, request
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

app = Flask(__name__)

# Initialize the object detection pipeline
object_detector = pipeline("object-detection", "facebook/detr-resnet-50")

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

def process_user(user, obj_name):
    """
    Process a user's profile

    Fetch the user's profile image, and use object detection model to detect
    objects in the image, and find any object whose label is the same as
    obj_name.

    Parameters:
    user: A dict, has details about the user's profile.
    obj_name: A string, the name of a object, eg.: `person` `dog` etc.
    """
    res = {
        "user_id": user['user_id'],
        "display_name": user['display_name'],
        "profile_image": user['profile_image'],
        "object_detected": False,
        "bounding_boxes": [],
        "detection_time_ms": 0,
    }

    profile_image_url = user['profile_image']
    if profile_image_url is None or profile_image_url.strip() == '':
        return res

    profile_image = fetch_profile_image(user['profile_image'])

    start_time = time.time()
    if profile_image is not None:
        detection_boxes = detect_object(profile_image, obj_name)
    else:
        detection_boxes = []
    end_time = time.time()

    latency = (end_time - start_time) * 1000

    res['object_detected'] = len(detection_boxes) > 0
    res['bounding_boxes'] = detection_boxes
    res['detection_time_ms'] = latency

    return res

@app.route('/api/v1/users', methods=['POST'])
def detections():
    data = request.get_json()

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
        res = process_user(user, object_name)
        detection_results.append(res)

    return jsonify(detection_results), 201

if __name__ == '__main__':
    app.run(debug=True)
