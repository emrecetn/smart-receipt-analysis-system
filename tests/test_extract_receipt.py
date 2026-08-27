from unittest.mock import MagicMock, patch

from openai import OpenAIError

import main
from conftest import fake_openai_response, fake_yolo_result

ENDPOINT = "/api/v1/extract-receipt"


# --- Kimlik dogrulama ---

def test_missing_api_key_returns_422(client, sample_image_bytes):
    resp = client.post(ENDPOINT, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})
    assert resp.status_code == 422


def test_invalid_api_key_returns_403(client, sample_image_bytes):
    resp = client.post(
        ENDPOINT,
        headers={"X-API-Key": "sk_gecersiz_bir_anahtar"},
        files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 403


def test_hash_api_key_is_deterministic_sha256():
    import hashlib

    raw = "sk_live_ornek_anahtar"
    assert main.hash_api_key(raw) == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_live_key_lookup_queries_by_hash_not_raw_value(client, sample_image_bytes):
    """Regression lock: validation must be done by hash, not the raw
    api_key. It used to compare against the raw text — this guarantees
    the raw key is never looked up in the database."""
    fake_supabase = MagicMock()
    fake_supabase.from_.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    raw_key = "sk_live_regresyon_testi"

    with patch.object(main, "supabase", fake_supabase):
        client.post(
            ENDPOINT,
            headers={"X-API-Key": raw_key},
            files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")},
        )

    eq_call_args = fake_supabase.from_.return_value.select.return_value.eq.call_args
    assert eq_call_args.args[0] == "api_key_hash"
    assert eq_call_args.args[1] == main.hash_api_key(raw_key)
    assert eq_call_args.args[1] != raw_key


def test_live_key_not_found_in_db_returns_403(client, sample_image_bytes):
    fake_supabase = MagicMock()
    fake_supabase.from_.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    with patch.object(main, "supabase", fake_supabase):
        resp = client.post(
            ENDPOINT,
            headers={"X-API-Key": "sk_live_veritabaninda_yok"},
            files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert resp.status_code == 403


def test_live_key_found_in_db_succeeds_auth(client, sample_image_bytes):
    fake_supabase = MagicMock()
    fake_supabase.from_.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        {"company_name": "Test A.Ş."}
    ]
    payload = {"merchant_name": "X", "tax_breakdown": []}
    with patch.object(main, "supabase", fake_supabase), \
         patch.object(main, "yolo_model", side_effect=lambda *a, **k: fake_yolo_result(detected=False)), \
         patch.object(main.client.chat.completions, "create", return_value=fake_openai_response(payload)):
        resp = client.post(
            ENDPOINT,
            headers={"X-API-Key": "sk_live_gercek_musteri"},
            files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert resp.status_code == 200


def test_live_key_without_supabase_connection_returns_503(client, sample_image_bytes):
    with patch.object(main, "supabase", None):
        resp = client.post(
            ENDPOINT,
            headers={"X-API-Key": "sk_live_her_hangi_bir_key"},
            files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert resp.status_code == 503


# --- Dosya dogrulama ---

def test_empty_file_returns_400(client, demo_headers):
    resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400


def test_oversized_file_returns_413(client, demo_headers):
    big_content = b"0" * (main.MAX_FILE_SIZE_BYTES + 1)
    resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("big.jpg", big_content, "image/jpeg")})
    assert resp.status_code == 413


def test_corrupt_image_returns_400(client, demo_headers):
    resp = client.post(
        ENDPOINT,
        headers=demo_headers,
        files={"file": ("bad.jpg", b"bu bir resim degil, sadece duz metin", "image/jpeg")},
    )
    assert resp.status_code == 400


# --- YOLO -> GPT-4o pipeline (mock'lu) ---

def test_successful_extraction_with_yolo_detection(client, demo_headers, sample_image_bytes):
    payload = {
        "merchant_name": "TEST MARKET",
        "tax_id": "1234567890",
        "date": "01.01.2026",
        "receipt_no": "42",
        "total_amount": 100.5,
        "tax_amount": 18.0,
        "tax_breakdown": [{"rate": 20, "amount": 18.0}],
    }
    with patch.object(main, "yolo_model", side_effect=lambda *a, **k: fake_yolo_result(detected=True)), \
         patch.object(main.client.chat.completions, "create", return_value=fake_openai_response(payload)):
        resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["yolo_detected"] is True
    assert body["data"]["merchant_name"] == "TEST MARKET"
    assert body["data"]["tax_breakdown"][0]["rate"] == 20
    assert body["cropped_image_base64"]
    assert set(body["durations"].keys()) == {"yolo_seconds", "openai_seconds", "total_seconds"}


def test_successful_extraction_without_detection_uses_original_image(client, demo_headers, sample_image_bytes):
    payload = {"merchant_name": None, "tax_breakdown": []}
    with patch.object(main, "yolo_model", side_effect=lambda *a, **k: fake_yolo_result(detected=False)), \
         patch.object(main.client.chat.completions, "create", return_value=fake_openai_response(payload)):
        resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})

    assert resp.status_code == 200
    assert resp.json()["yolo_detected"] is False


def test_yolo_exception_returns_500(client, demo_headers, sample_image_bytes):
    with patch.object(main, "yolo_model", side_effect=RuntimeError("model çöktü")):
        resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})
    assert resp.status_code == 500


def test_openai_error_returns_502(client, demo_headers, sample_image_bytes):
    with patch.object(main, "yolo_model", side_effect=lambda *a, **k: fake_yolo_result(detected=False)), \
         patch.object(main.client.chat.completions, "create", side_effect=OpenAIError("bağlantı hatası")):
        resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})
    assert resp.status_code == 502


def test_malformed_json_from_model_returns_502(client, demo_headers, sample_image_bytes):
    broken_response = fake_openai_response({})
    broken_response.choices[0].message.content = "bu gecerli bir json degil {{{"
    with patch.object(main, "yolo_model", side_effect=lambda *a, **k: fake_yolo_result(detected=False)), \
         patch.object(main.client.chat.completions, "create", return_value=broken_response):
        resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})
    assert resp.status_code == 502


def test_schema_mismatch_from_model_returns_502(client, demo_headers, sample_image_bytes):
    """Even if the model returns syntactically valid JSON, a schema mismatch
    must result in a 502, not be left to FastAPI's automatic response_model
    validation and fall through as a 500."""
    payload = {"tax_breakdown": [{"rate": "yirmi", "amount": "cok"}]}
    with patch.object(main, "yolo_model", side_effect=lambda *a, **k: fake_yolo_result(detected=False)), \
         patch.object(main.client.chat.completions, "create", return_value=fake_openai_response(payload)):
        resp = client.post(ENDPOINT, headers=demo_headers, files={"file": ("r.jpg", sample_image_bytes, "image/jpeg")})
    assert resp.status_code == 502
