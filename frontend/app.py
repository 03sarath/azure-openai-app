"""
Frontend — thin Flask UI layer
-------------------------------
This app ONLY serves the HTML page and proxies API calls to the backend container.
It contains zero RAG or AI logic — that all lives in backend/api.py.

Environment variable:
  BACKEND_URL  — set to http://backend:8000 inside Docker (via docker-compose)
                 defaults to http://localhost:8000 for local development
"""

import os
import requests
from requests.exceptions import ConnectionError, Timeout
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Reusable error response when the backend is not yet reachable
def backend_unavailable():
    return jsonify({"error": "Backend is still starting up. Please wait a moment and refresh."}), 503


# ── Page ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


# ── API proxy routes ──────────────────────────────────────────────────────
# Every route wraps the backend call in try/except so that if the backend
# is briefly unreachable (e.g. still building the FAISS index) the browser
# gets a clean JSON error instead of a 500 traceback.

@app.route("/api/health")
def health():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


@app.route("/api/documents")
def list_documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


@app.route("/api/documents/add", methods=["POST"])
def add_document():
    try:
        file = request.files["file"]
        resp = requests.post(
            f"{BACKEND_URL}/documents/add",
            files={"file": (file.filename, file.stream, file.content_type)},
            timeout=60,  # indexing can take a few seconds
        )
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


@app.route("/api/documents/delete/<filename>", methods=["DELETE"])
def delete_document(filename: str):
    try:
        resp = requests.delete(f"{BACKEND_URL}/documents/delete/{filename}", timeout=30)
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


@app.route("/api/documents/reindex/<filename>", methods=["POST"])
def reindex_document(filename: str):
    try:
        resp = requests.post(f"{BACKEND_URL}/documents/reindex/{filename}", timeout=60)
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        resp = requests.post(f"{BACKEND_URL}/ask", json=request.get_json(), timeout=60)
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    try:
        resp = requests.post(f"{BACKEND_URL}/history/clear", timeout=5)
        return jsonify(resp.json()), resp.status_code
    except (ConnectionError, Timeout):
        return backend_unavailable()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
