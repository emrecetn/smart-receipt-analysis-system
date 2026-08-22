def test_health_check_returns_ok_shape(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["yolo_loaded"] is True
    assert isinstance(body["supabase_connected"], bool)


def test_root_serves_html_and_is_hidden_from_schema(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
