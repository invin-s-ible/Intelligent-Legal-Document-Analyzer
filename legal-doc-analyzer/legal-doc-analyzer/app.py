"""Flask app: python app.py  ->  http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request
from analyzer import LegalAnalyzer
from analyzer.readers import read_upload

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
analyzer = LegalAnalyzer()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/extract")
def extract():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file received."), 400
    try:
        return jsonify(text=read_upload(f))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/analyze")
def analyze():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text.split()) < 15:
        return jsonify(error="Add at least a few sentences of text."), 400
    return jsonify(analyzer.analyze(text))


if __name__ == "__main__":
    app.run(debug=True)
