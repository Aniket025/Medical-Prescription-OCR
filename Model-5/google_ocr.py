from __future__ import print_function
from base64 import b64encode
from os import makedirs, remove
from os.path import join, basename
from sys import argv
import json
import requests
import glob
from unidecode import unidecode

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
RESULTS_DIR = 'jsons'

def make_image_data_list(image_filenames):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    with open(image_filenames, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_requests.append({
                'image': {'content': ctxt},
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }]
        })
    return img_requests

def make_image_data(image_filenames):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filenames)
    return json.dumps({"requests": imgdict }).encode()


def request_ocr(api_key, image_filenames):
    response = requests.post(ENDPOINT_URL, data=make_image_data(image_filenames), params={'key': api_key}, headers={'Content-Type': 'application/json'})
    return response

def remove_non_ascii(text):
    return unidecode(unicode(text, encoding = "utf-8"))

def convert(filename):
    api_key = "AIzaSyCSMpzBIKlZObk8Uzkx6Iavo967m7vFf0Q"
    image_filename = filename
    entities = []
    bBoxes = []
    if not api_key or not image_filename:
        print("""Please supply a valid Google Cloud Vision api key. Follow this link for details""")
        print("""https://cloud.google.com/vision/""")
    else:
        response = request_ocr(api_key, image_filename)
        if response.status_code != 200 or response.json().get('error'):
            print(response.text)
        else:
            for i in range(1,len(response.json()['responses'][0]['textAnnotations'])):
                entities.append(remove_non_ascii(response.json()['responses'][0]['textAnnotations'][i]['description'].encode("utf-8")))
            for i in range(1,len(response.json()['responses'][0]['textAnnotations'])):
                bBoxes.append(response.json()['responses'][0]['textAnnotations'][i]['boundingPoly']['vertices'])
    return entities,bBoxes

if __name__ == "__main__":
    convert(argv[1])
