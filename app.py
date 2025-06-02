from flask import Flask, render_template, request, jsonify
import threading

app = Flask(__name__)

# In-memory storage for transcriptions and Gemini outputs
transcriptions = []
gemini_outputs = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_transcription', methods=['POST'])
def add_transcription():
    data = request.json
    if 'text' in data:
        transcriptions.append(data['text'])
        return jsonify({'status': 'success', 'message': 'Transcription added'}), 200
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

@app.route('/add_gemini_output', methods=['POST'])
def add_gemini_output():
    data = request.json
    if 'text' in data:
        gemini_outputs.append(data['text'])
        return jsonify({'status': 'success', 'message': 'Gemini output added'}), 200
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

@app.route('/get_transcriptions')
def get_transcriptions():
    return jsonify({'transcriptions': transcriptions})

@app.route('/get_gemini_outputs')
def get_gemini_outputs():
    return jsonify({'gemini_outputs': gemini_outputs})

def run_flask_app():
    # Run Flask in a separate thread or use a production-ready server like gunicorn/waitress
    # For simplicity in development, using Flask's built-in server with debug=False
    # and threaded=True to handle multiple requests and not block main script
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    # This is for running the Flask app directly for testing the UI standalone
    print("Running Flask app directly for UI testing...")
    app.run(host='0.0.0.0', port=5000, debug=True) 