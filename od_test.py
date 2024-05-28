import unittest

from PIL import Image
from unittest.mock import patch, Mock
from flask import json

import od

class TestObjectDetectionAPI(unittest.TestCase):

    def setUp(self):
        self.app = od.app.test_client()
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
        # Mock a failed image fetch
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = od.fetch_profile_image('http://example.com/image.png')
        self.assertIsNone(result)

    @patch('od.requests.get')
    def test_fetch_so_top_users_success(self, mock_get):
        # Mock a successful fetch from StackOverflow API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'items': [{'user_id': 1, 'display_name': 'User1', 'profile_image': 'http://example.com/image.png'}]}
        mock_get.return_value = mock_response

        result = od.fetch_so_top_users()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['user_id'], 1)

    @patch('od.requests.get')
    def test_fetch_so_top_users_failure(self, mock_get):
        # Mock a failed fetch from StackOverflow API
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with self.assertRaises(od.ProxyError):
            od.fetch_so_top_users()

    @patch('od.object_detector')
    def test_detect_object(self, mock_object_detector):
        # Mock the object detection pipeline
        mock_object_detector.return_value = [{'label': 'person', 'box': [0, 0, 100, 100]}]

        image = Image.new('RGB', (100, 100))
        result = od.detect_object(image, 'person')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'person')

    @patch('od.fetch_profile_image')
    @patch('od.fetch_so_top_users')
    @patch('od.object_detector')
    def test_detections_endpoint(self, mock_object_detector, mock_fetch_so_top_users, mock_fetch_profile_image):
        # Mock the dependencies for the endpoint
        mock_fetch_so_top_users.return_value = [{'user_id': 1, 'display_name': 'User1', 'profile_image': 'http://example.com/image.png'}]
        mock_fetch_profile_image.return_value = Image.new('RGB', (100, 100))
        mock_object_detector.return_value = [{'label': 'person', 'box': [0, 0, 100, 100]}]

        # Call the endpoint
        response = self.app.post('/api/v1/users', data=json.dumps({
            'query': {'object': 'person'}
        }), content_type='application/json')

        # Check the response
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['user_id'], 1)
        self.assertTrue(data[0]['object_detected'])

if __name__ == '__main__':
    unittest.main()

