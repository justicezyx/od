import od
import unittest

from unittest.mock import patch, Mock
from io import BytesIO
from PIL import Image

from od import app

class TestOD(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    @patch('od.requests.get')
    def test_fetch_profile_image_success(self, mock_get):
        # Mock a successful image fetch
        mock_response = Mock()
        mock_response.status_code = 200
        with open('./test.PNG', 'rb') as file:
            mock_response.content = file.read()
        mock_get.return_value = mock_response

        result = od.fetch_profile_image('http://example.com/image.png')
        self.assertIsInstance(result, Image.Image)

    @patch('od.requests.get')
    def test_fetch_profile_image_failure(self, mock_get):
        # Mock a failed response from requests.get
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        image = od.fetch_profile_image('http://example.com/image.png')
        self.assertIsNone(image)

    def test_not_whitespace_string(self):
        self.assertTrue(od.not_whitespace_string('   '))
        with self.assertRaises(ValueError):
            od.not_whitespace_string(123)

    @patch('od.requests.get')
    def test_fetch_so_top_users_success(self, mock_get):
        # Mock a successful response from requests.get
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'items': [{'user_id': 1, 'display_name': 'Test User', 'profile_image': 'http://example.com/image.png'}]}
        mock_get.return_value = mock_response

        users = od.fetch_so_top_users()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]['user_id'], 1)

    @patch('od.requests.get')
    def test_fetch_so_top_users_failure(self, mock_get):
        # Mock a failed response from requests.get
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with self.assertRaises(od.ProxyError):
            od.fetch_so_top_users()

    @patch('od.object_detector')
    def test_detect_object(self, mock_object_detector):
        # Mock the object detection pipeline
        mock_object_detector.return_value = [{'label': 'person', 'box': [1, 2, 3, 4]}]
        image = Image.new('RGB', (60, 30), color = 'red')
        boxes = od.detect_object(image, 'person')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['label'], 'person')

    @patch('od.fetch_profile_image')
    @patch('od.detect_object')
    def test_process_user(self, mock_detect_object, mock_fetch_profile_image):
        mock_detect_object.return_value = [{'label': 'person', 'box': [1, 2, 3, 4]}]
        image = Image.new('RGB', (60, 30), color = 'red')
        mock_fetch_profile_image.return_value = image

        user = {'user_id': 1, 'display_name': 'Test User', 'profile_image': 'http://example.com/image.png'}
        result = od.process_user(user, 'person')

        self.assertTrue(result['object_detected'])
        self.assertEqual(len(result['bounding_boxes']), 1)
        self.assertGreater(result['detection_time_ms'], 0)

    @patch('od.fetch_so_top_users')
    @patch('od.process_user')
    def test_detections(self, mock_process_user, mock_fetch_so_top_users):
        mock_fetch_so_top_users.return_value = [{'user_id': 1, 'display_name': 'Test User', 'profile_image': 'http://example.com/image.png'}]
        mock_process_user.return_value = {'user_id': 1, 'display_name': 'Test User', 'profile_image': 'http://example.com/image.png', 'object_detected': True, 'bounding_boxes': [], 'detection_time_ms': 100}

        response = self.app.post('/api/v1/users', json={'query': {'object': 'person'}})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.json), 1)
        self.assertTrue(response.json[0]['object_detected'])

    def test_detections_missing_query(self):
        response = self.app.post('/api/v1/users', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(response.json['error'], "Query parameter 'query' is required")

    def test_detections_missing_object_field(self):
        response = self.app.post('/api/v1/users', json={'query': {}})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(response.json['error'], "Field 'object' not found in the query")

    def test_detections_object_not_string(self):
        response = self.app.post('/api/v1/users', json={'query': {'object': 123}})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(response.json['error'], "Field 'object' is not a string")

    def test_detections_object_whitespace(self):
        response = self.app.post('/api/v1/users', json={'query': {'object': '   '}})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(response.json['error'], "Field 'object' is white space")

if __name__ == '__main__':
    unittest.main()

