import io

from PIL import Image

import main


def _jpeg_bytes(w, h, color=(10, 20, 30)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(w, h, color=(200, 100, 50)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _heic_bytes(w, h):
    import pillow_heif

    img = Image.new("RGB", (w, h), (5, 100, 200))
    heif_file = pillow_heif.from_pillow(img)
    buf = io.BytesIO()
    heif_file.save(buf, format="HEIF")
    return buf.getvalue()


def test_decode_valid_jpeg_returns_correct_shape():
    result = main.decode_image(_jpeg_bytes(64, 48))
    assert result is not None
    assert result.shape[:2] == (48, 64)  # (height, width)


def test_decode_valid_png_returns_correct_shape():
    result = main.decode_image(_png_bytes(30, 40))
    assert result is not None
    assert result.shape[:2] == (40, 30)


def test_decode_heic_falls_back_to_pillow():
    """cv2.imdecode HEIC'i desteklemiyor; Pillow fallback devreye girmeli."""
    result = main.decode_image(_heic_bytes(32, 24))
    assert result is not None
    assert result.shape[:2] == (24, 32)


def test_decode_garbage_bytes_returns_none():
    assert main.decode_image(b"bu bir gorsel degil, sadece duz metin") is None


def test_decode_empty_bytes_returns_none():
    assert main.decode_image(b"") is None
