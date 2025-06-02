# Copyright 2024 Rich Cannings <rcannings@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

@app.route('/clear_transcriptions', methods=['POST'])
def clear_transcriptions_route():
    global transcriptions
    transcriptions = []
    return jsonify({'status': 'success', 'message': 'Transcriptions cleared'}), 200

@app.route('/clear_gemini_outputs', methods=['POST'])
def clear_gemini_outputs_route():
    global gemini_outputs
    gemini_outputs = []
    return jsonify({'status': 'success', 'message': 'Gemini outputs cleared'}), 200

@app.route('/remove_detected_call_sign', methods=['POST'])
def remove_detected_call_sign_route():
    global gemini_outputs
    data = request.json
    text_to_remove = data.get('text')

    if not text_to_remove:
        return jsonify({'status': 'error', 'message': 'Invalid data: text to remove is missing'}), 400

    try:
        gemini_outputs.remove(text_to_remove)
        return jsonify({'status': 'success', 'message': 'Detected call sign removed'}), 200
    except ValueError:
        # This means the item was not found in the list, which could happen
        # if it was already removed or if there's a mismatch.
        # For client-side removal, this might not be a critical error if already removed from view.
        print(f"Attempted to remove non-existent item from gemini_outputs: {text_to_remove}")
        return jsonify({'status': 'warning', 'message': 'Item not found or already removed'}), 202 # 202 Accepted, but no action taken
    except Exception as e:
        print(f"Error removing detected call sign: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

def run_flask_app():
    # Run Flask in a separate thread or use a production-ready server like gunicorn/waitress
    # For simplicity in development, using Flask's built-in server with debug=False
    # and threaded=True to handle multiple requests and not block main script
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    # This is for running the Flask app directly for testing the UI standalone
    print("Running Flask app directly for UI testing...")
    app.run(host='0.0.0.0', port=5000, debug=True) 