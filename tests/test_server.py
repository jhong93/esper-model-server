import os
import pytest
import random
import requests
import time
from subprocess import Popen, PIPE

PORT = random.randint(20000, 30000)
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(FILE_DIR, '..')
MODEL_DATA_DIR = os.path.join(ROOT_DIR, 'data')
TEST_IMG_PATH = os.path.join(FILE_DIR, 'test.jpg')


@pytest.fixture('session', autouse=True)
def run_server():
    print('Starting server on port {}'.format(PORT))
    p = Popen([
        'python3', 'server.py', '--model-dir', MODEL_DATA_DIR,
        '-p', str(PORT)
    ], stdout=PIPE)
    print('Waiting for server to load...')
    while True:
        line = p.stdout.readline()
        if line == b'':
            raise Exception('Failed to start server')
        if 'Serving Flask app' in line.decode('ascii'):
            break
    print('Server loaded')
    time.sleep(2)
    yield
    p.kill()


def test_face_detect():
    url = 'http://localhost:{}/face-detect'.format(PORT)
    res = requests.get(url, params={'path': TEST_IMG_PATH}).json()
    assert len(res[TEST_IMG_PATH]) == 2, 'Expected 2 faces'
    res2 = requests.get(url, params={'path': FILE_DIR}).json()
    assert res == res2, 'Single and batch results do not match'


def test_face_embed():
    url = 'http://localhost:{}/face-embed'.format(PORT)
    emb = requests.get(url, params={
        'path': TEST_IMG_PATH, 'x1': 100, 'x2': 200, 'y1': 100, 'y2': 200
    }).json()
    assert len(emb) == 128, 'Incorrect embedding dimension'
