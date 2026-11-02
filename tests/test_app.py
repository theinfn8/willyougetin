"""Flask routes, using a fake predictor (no model needed)."""

import pytest

from service.app import create_app


class FakePredictor:  # pylint: disable=too-few-public-methods
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def predict(self, applicant):
        self.calls.append(applicant)
        if self.fail:
            raise RuntimeError("boom: /secret/path")
        return {"label": "Accepted", "p_accepted": 0.734, "confidence": 0.734,
                "input_text": "Program: Physics\nComments: <b>hi</b>"}


FORM = {"program": "Physics", "degree": "PhD", "citizenship": "American",
        "gpa": "3.9", "comments": "<script>alert(1)</script>"}


@pytest.fixture
def fake():
    return FakePredictor()


@pytest.fixture
def client(fake):
    return create_app(predictor=fake).test_client()


def test_form_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Will You Get In?" in response.data
    assert b"not admissions advice" in response.data


def test_valid_post_shows_prediction(client, fake):
    response = client.post("/", data=FORM)
    assert response.status_code == 200
    assert b"Prediction: Accepted" in response.data
    assert b"73% chance" in response.data
    assert fake.calls[0]["GPA"] == 3.9


def test_user_text_is_escaped(client):
    body = client.post("/", data=FORM).data
    assert b"<script>alert(1)</script>" not in body
    assert b"&lt;script&gt;" in body
    assert b"<b>hi</b>" not in body


def test_invalid_post_shows_errors_without_predicting(client, fake):
    response = client.post("/", data={**FORM, "gre": "999", "degree": "MBA"})
    assert response.status_code == 400
    assert b"GRE Quant must be between 130 and 170" in response.data
    assert b"Please choose a degree" in response.data
    assert b'value="999"' in response.data  # the visitor's input is kept
    assert not fake.calls


def test_prediction_failure_is_generic():
    client = create_app(predictor=FakePredictor(fail=True)).test_client()
    response = client.post("/", data=FORM)
    assert response.status_code == 500
    assert b"Something went wrong" in response.data
    assert b"/secret/path" not in response.data


def test_missing_model():
    client = create_app(load_model=False).test_client()
    assert client.post("/", data=FORM).status_code == 503
    health = client.get("/health")
    assert health.status_code == 503
    assert health.get_json() == {"status": "unavailable", "model_loaded": False}


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_loaded": True}


def test_security_headers(client):
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_urls_keep_the_proxy_prefix(client):
    body = client.get("/", headers={"X-Forwarded-Prefix": "/willyougetin"}).data
    assert b'action="/willyougetin/"' in body
    assert b'href="/willyougetin/static/style.css"' in body
    assert b'src="/willyougetin/static/predict.js"' in body


def test_oversized_request_is_refused(client):
    assert client.post("/", data={**FORM, "comments": "x" * 20000}).status_code == 413
