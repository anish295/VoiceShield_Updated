#!/usr/bin/env python3
"""
Simple HTTP Server for VoiceShield
"""

from flask import Flask, render_template, jsonify, send_from_directory, redirect, request, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import os
import json
import base64
import time
import engineio

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Routes
@app.route('/')
def index():
    """Redirect to the updated intro page"""
    return redirect('/intro.html')

@app.route('/api/status')
def status():
    """API endpoint to check server status"""
    response = jsonify({"status": "online"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Content-Type', 'application/json')
    return response

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Mock API endpoint for audio/facial analysis"""
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    response = jsonify({
        "status": "success",
        "emotion": "neutral",
        "confidence": 0.85,
        "message": "Analysis completed successfully"
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Content-Type', 'application/json')
    return response

@app.route('/api/emotion', methods=['POST', 'OPTIONS'])
def emotion():
    """Mock API endpoint for emotion detection"""
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    response = jsonify({
        "emotion": "neutral",
        "confidence": 0.85
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Content-Type', 'application/json')
    return response

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    # Mock processing the frame
    socketio.emit('processed_frame', {'frame': data.get('frame', '')})
    # Send mock emotion update
    socketio.emit('emotion_update', {
        'overall': {'emotion': 'neutral', 'confidence': 0.8},
        'facial': [{'emotion': 'neutral', 'confidence': 0.75}],
        'voice': [{'emotion': 'neutral', 'confidence': 0.85, 'source': 'real_audio'}]
    })

@socketio.on('audio_data')
def handle_audio(data):
    # Mock processing audio data
    pass

# Serve static files from the project's frontend directory (not the backend/frontend)
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the updated frontend directory"""
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
    return send_from_directory(frontend_dir, path)

if __name__ == '__main__':
    print("Starting VoiceShield simple server...")
    print("Server running at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)