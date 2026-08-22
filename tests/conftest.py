import io
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main

DEMO_HEADERS = {"X-API-Key": "sk_test_demo123456789"}


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def demo_headers():
    return dict(DEMO_HEADERS)


def make_jpeg_bytes(width=200, height=300, color=(255, 255, 255)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_image_bytes():
    return make_jpeg_bytes()


def fake_yolo_result(detected: bool, x1=10, y1=10, x2=100, y2=200):
    """ultralytics'in Results objesini taklit eden sahte bir sonuc uretir."""
    fake_boxes = MagicMock()
    if detected:
        fake_box = MagicMock()
        fake_box.xyxy = [[x1, y1, x2, y2]]
        fake_boxes.__len__.return_value = 1
        fake_boxes.__getitem__.return_value = fake_box
    else:
        fake_boxes.__len__.return_value = 0
    fake_result = MagicMock()
    fake_result.boxes = fake_boxes
    return [fake_result]


def fake_openai_response(payload: dict):
    """OpenAI ChatCompletion yanitini taklit eder."""
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response
