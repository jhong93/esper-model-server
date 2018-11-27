#!/usr/bin/env python3

import argparse
import cv2
import math
import numpy as np
import os
import pathlib
from collections import namedtuple
from flask import Flask, request, abort, jsonify

import models.facenet as facenet
import models.mtcnn as mtcnn


DEFAULT_PORT = 9999


def get_args():
    parser = argparse.ArgumentParser('Serve some models')
    parser.add_argument('--model-dir', dest='model_dir', type=str,
                        default='data')
    parser.add_argument('-p', '--port', type=int, default=DEFAULT_PORT,
                        help='Port number. Default: {}'.format(DEFAULT_PORT))
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Run in debug mode')
    return parser.parse_args()


Models = namedtuple('Models', ['face_detector', 'face_embeddor'])

IMG_EXTS = {'.jpg', '.png'}
KB = 1024


def file_size(filename):
    st = os.stat(filename)
    return st.st_size


def load_models(model_dir):
    face_embeddor = facenet.FaceNetEmbed(os.path.join(model_dir, 'facenet'))
    face_detector = mtcnn.MTCNN(os.path.join(model_dir, 'align'))
    return Models(face_detector=face_detector, face_embeddor=face_embeddor)


def build_app(models):
    app = Flask(__name__)

    @app.route('/face-detect')
    def face_detect():
        path = request.args.get('path')

        raw_images = []
        img_paths = []
        if os.path.isdir(path):
            print('Directory: {}'.format(path))
            for img in os.listdir(path):
                if pathlib.Path(img).suffix.lower() not in IMG_EXTS:
                    continue
                img_path = os.path.join(path, img)
                if not os.path.isfile(img_path):
                    continue
                if file_size(img_path) < 5 * KB:
                    continue
                im = cv2.imread(img_path)
                if im is None:
                    continue
                raw_images.append(im)
                img_paths.append(img_path)
        elif os.path.isfile(path):
            print('File: {}'.format(path))
            im = cv2.imread(path)
            if im is None:
                abort(404)
            raw_images.append(im)
            img_paths.append(path)
        else:
            abort(400)

        detected_faces = models.face_detector.face_detect(raw_images)
        result = {}
        for img_path, detections in zip(img_paths, detected_faces):
            bboxes = []
            for box in detections:
                x1 = int(math.floor(box.x1))
                x2 = int(math.ceil(box.x2))
                y1 = int(math.floor(box.y1))
                y2 = int(math.ceil(box.y2))
                if x1 < x2 and y1 < y2:
                    bboxes.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
            if bboxes:
                result[img_path] = bboxes
        return jsonify(result)

    @app.route('/face-embed', methods=['GET', 'POST'])
    def face_embed():
        if request.method == 'POST':
            width = int(request.args.get('width'))
            height = int(request.args.get('height'))
            img_bin = request.get_data()
            if len(img_bin) == 0:
                abort(400)
            img = np.fromstring(img_bin, dtype=np.uint8).reshape(
                (height, width, 3))
        elif request.method == 'GET' and 'path' in request.args:
            img_path = request.args.get('path')
            x1 = int(request.args.get('x1'))
            x2 = int(request.args.get('x2'))
            y1 = int(request.args.get('y1'))
            y2 = int(request.args.get('y2'))
            img = cv2.imread(img_path)
            img = img[y1:y2, x1:x2, :]
        else:
            abort(400)
        assert img.size > 0
        emb = models.face_embeddor.embed(img)
        return jsonify([float(x) for x in emb.tolist()])

    @app.route('/')
    def root():
        return 'Model Server'

    return app


def main(model_dir, port, debug):
    models = load_models(model_dir)
    app = build_app(models)
    server_args = {}
    server_args['debug'] = debug
    server_args['host'] = 'localhost'
    server_args['port'] = port
    app.run(**server_args)


if __name__ == '__main__':
    main(**vars(get_args()))
