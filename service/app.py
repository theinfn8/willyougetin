"""Flask app for the "Will You Get In?" demo.

Routes (relative to wherever the app is mounted, e.g. ``/willyougetin``):

* ``GET  /``        -- the form
* ``POST /``        -- validate, predict, and show the result above the form
* ``GET  /health``  -- 200 when the model is loaded, 503 otherwise

Configuration (environment variables):

* ``MODEL_DIR``   -- directory with the packaged model
  (default ``service/model_files``, filled by ``scripts/package_model.sh``)
* ``ORT_THREADS`` -- onnxruntime threads per prediction (default 1)
"""

import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .validation import CITIZENSHIPS, DEGREES, DEFAULTS, NUMERIC_FIELDS, validate

log = logging.getLogger(__name__)

SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_predictor(model_dir, threads):
    """Load the model, or return ``(None, reason)`` so the app can still serve."""
    # Imported here so tests can run the app without onnxruntime installed.
    from .predictor import Predictor  # pylint: disable=import-outside-toplevel
    try:
        return Predictor(model_dir, threads=threads), None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Any failure (missing files, corrupt model) means "model unavailable";
        # the page says so instead of crashing, and /health reports 503.
        log.exception("Could not load model from %s", model_dir)
        return None, str(exc)


def create_app(predictor=None, load_model=True):
    """Build the app. Pass ``predictor`` to inject a fake one in tests."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # the form is tiny

    # Apache forwards /willyougetin/* here and sends X-Forwarded-Prefix, so
    # generated URLs (form action, static files) keep the prefix. Without
    # the headers, e.g. running locally, nothing changes.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    if predictor is None and load_model:
        predictor, _ = load_predictor(
            os.environ.get("MODEL_DIR", os.path.join(SERVICE_ROOT, "model_files")),
            int(os.environ.get("ORT_THREADS", "1")),
        )

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.route("/", methods=["GET", "POST"])
    def index():
        values, prediction, errors = dict(DEFAULTS), None, {}
        status = 200
        if request.method == "POST":
            values, applicant, errors = validate(request.form)
            if errors:
                status = 400
            elif predictor is None:
                errors = {"form": "The prediction model isn't available right "
                                  "now. Please try again later."}
                status = 503
            else:
                try:
                    prediction = predictor.predict(applicant)
                except Exception:  # pylint: disable=broad-exception-caught
                    log.exception("Prediction failed")
                    errors = {"form": "Something went wrong while making a "
                                      "prediction. Please try again."}
                    status = 500
        return render_template(
            "index.html", values=values, prediction=prediction, errors=errors,
            degrees=DEGREES, citizenships=CITIZENSHIPS, numeric=NUMERIC_FIELDS,
        ), status

    @app.route("/health")
    def health():
        if predictor is None:
            # The reason (load_error) is in the log, not the public response:
            # it can contain server file paths.
            return jsonify(status="unavailable", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True)

    return app
