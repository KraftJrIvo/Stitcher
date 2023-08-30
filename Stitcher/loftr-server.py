#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import cv2
import base64
import numpy as np

import kornia as K
import kornia.feature as KF
import numpy as np
import torch

import json

def decode_from_bin(bin_data):
    padding = len(bin_data) % 4
    if padding > 0:
        bin_data += b'='* (4 - padding)
    bin_data = base64.b64decode(bin_data)
    image = np.asarray(bytearray(bin_data), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    h = int(img.shape[0])
    w = int(img.shape[1] / 2)
    left = img[0:h, 0:w]
    right = img[0:h, w:(w * 2)]

    return [left, right]

def encode_from_cv2(img):
    bin = cv2.imencode('.jpg', img)[1]
    return str(base64.b64encode(bin), "utf-8")

def load_torch_image(imgin):
    img = K.image_to_tensor(imgin, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        #logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        imgs = decode_from_bin(post_data)

        #im1re = cv2.resize(imgs[0], (700, 700))  
        #im2re = cv2.resize(imgs[1], (700, 700))      
        img1 = load_torch_image(imgs[0])
        img2 = load_torch_image(imgs[1])
        matcher = KF.LoFTR(pretrained='outdoor')
        input_dict = {"image0": K.color.rgb_to_grayscale(img1), "image1": K.color.rgb_to_grayscale(img2)}
        with torch.no_grad():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
        result = cv2.hconcat([mkpts0, mkpts1])
        jsonString = json.dumps(result.tolist())
        
        self._set_response()
        self.wfile.write(jsonString.encode())

def run(server_class=HTTPServer, handler_class=S, port=8867):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    run()
