from __future__ import print_function
from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
import requests

class GoogleCloudVisionOCR(object):
    def __init__(self, filename):
        self.ENDPOINT_URL = "https://vision.googleapis.com/v1/images:annotate"
        self.RESULTS_DIR = 'jsons'
        self.api_key = "AIzaSyCSMpzBIKlZObk8Uzkx6Iavo967m7vFf0Q"
        self.filename = filename

    def make_image_data_list():
        """
        image_filename is the filename string
        Returns a list of dicts formatted as the Vision API needs them to be
        """
        img_requests = []
        with open(self.filename, 'rb') as f:
            ctxt = b64encode(f.read()).decode()
            img_requests.append({'image': {'content': ctxt},'features': [{'type': 'TEXT_DETECTION','maxResults': 1}]})
        return img_requests

    def make_image_data():
        """Returns the image data lists as bytes"""
        imgdict = make_image_data_list(self.filename)
        return json.dumps({"requests": imgdict }).encode()


    def request_ocr():
        response = requests.post(ENDPOINT_URL, data=make_image_data(self.filename), params={'key': self.api_key}, headers={'Content-Type': 'application/json'})
        return response
