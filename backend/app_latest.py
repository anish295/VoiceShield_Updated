#!/usr/bin/env python3
"""
VoiceShield WORKING backend (ported from Latest/backend/app.py) for Ayush Version.
Provides Socket.IO endpoints: /api/start, /api/stop, and events process_frame, audio_chunk.
"""

import os
import sys
import time
import base64
import logging
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace


current_dir = Path(__file__).parent
parent_dir = current_dir.parent

app = Flask(__name__, template_folder=str(parent_dir / 'frontend'), static_folder=str(parent_dir / 'frontend' / 'static'))
app.config['SECRET_KEY'] = 'voiceshield_secret_key'
CORS(app, resources={r"/*": {"origins": ["*"], "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


logger = logging.getLogger("voiceshield")
logging.basicConfig(level=logging.INFO)

is_running = False
face_cascade = None

current_emotions = {
    'facial': [],
    'voice': [],
    'overall': []
}

anger_config = {
    'threshold': 0.6,
    'cooldown': 30,
    'enabled': True
}


def decode_base64_image(base64_string: str):
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def initialize_face_detection() -> bool:
    global face_cascade
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("Could not load face cascade at: %s", cascade_path)
            return False
        try:
            from deepface import DeepFace  # noqa: F401
            logger.info("DeepFace is available for emotion detection.")
        except ImportError as e:
            logger.error(f"DeepFace not installed: {e}")
            return False
        return True
    except Exception as e:
        logger.error(f"Face detection initialization failed: {e}")
        return False


def detect_faces_and_emotions(frame):
    """Use DeepFace to analyze emotions; fallback to Haar if needed."""
    facial_emotions = []
    try:
        # DeepFace can handle detection + emotion
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # DeepFace may return list or dict depending on version
        results = analysis if isinstance(analysis, list) else [analysis]
        for r in results:
            region = r.get('region') or {}
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotions = r.get('emotion') or {}
            if emotions:
                dominant = r.get('dominant_emotion') or max(emotions, key=emotions.get)
                conf = float(emotions.get(dominant, 0.0)) / 100.0 if max(emotions.values()) > 1 else float(emotions.get(dominant, 0.0))
                facial_emotions.append({
                    'emotion': dominant,
                    'confidence': max(0.0, min(conf, 1.0)),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'all_scores': {k: (v/100.0 if max(emotions.values()) > 1 else float(v)) for k, v in emotions.items()}
                })
    except Exception as e:
        logger.error(f"DeepFace analysis failed, falling back: {e}")
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                facial_emotions.append({'emotion': 'neutral', 'confidence': 0.5, 'bbox': [int(x), int(y), int(w), int(h)]})
        except Exception as e2:
            logger.error(f"Fallback face detection failed: {e2}")
    return facial_emotions, frame


@app.route('/api/start', methods=['POST'])
def start_system():
    global is_running
    try:
        if not initialize_face_detection():
            return jsonify({'success': False, 'error': 'Face/DeepFace init failed'})
        is_running = True
        return jsonify({'success': True, 'audio_available': True, 'camera_available': True})
    except Exception as e:
        logger.error(f"System start failed: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stop', methods=['POST'])
def stop_system():
    global is_running
    is_running = False
    return jsonify({'success': True})


@app.route('/')
def serve_index():
    try:
        return send_from_directory(str(parent_dir / 'frontend'), 'index.html')
    except Exception:
        return jsonify({'error': 'index not found'}), 404

@app.route('/favicon.ico')
def favicon():
    return ('', 204)


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'camera_active': is_running,
        'face_detection_ready': face_cascade is not None,
        'audio_active': is_running,
        'system_running': is_running
    })


@app.route('/api/anger_alert/config', methods=['GET', 'POST'])
def anger_alert_config():
    global anger_config
    if request.method == 'GET':
        return jsonify(anger_config)
    try:
        data = request.get_json(force=True) or {}
        anger_config['threshold'] = float(data.get('threshold', anger_config['threshold']))
        anger_config['cooldown'] = int(data.get('cooldown', anger_config['cooldown']))
        anger_config['enabled'] = bool(data.get('enabled', anger_config['enabled']))
        return jsonify({'success': True, **anger_config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@socketio.on('process_frame')
def handle_process_frame(data):
    try:
        if not is_running:
            return
        base64_image = data.get('image')
        if not base64_image:
            return
        frame = decode_base64_image(base64_image)
        if frame is None:
            return
        facial_emotions, processed_frame = detect_faces_and_emotions(frame)
        current_emotions['facial'] = facial_emotions
        overall_emotion = {'emotion': facial_emotions[0]['emotion'] if facial_emotions else 'neutral', 'confidence': facial_emotions[0]['confidence'] if facial_emotions else 0.5, 'source': 'facial_primary' if facial_emotions else 'default'}
        current_emotions['overall'] = overall_emotion
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        emit('emotion_update', current_emotions)
        emit('processed_frame', {'frame': processed_frame_b64})
    except Exception as e:
        logger.error(f"Frame processing error: {e}")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    try:
        # Heuristic voice emotion using energy and zero-crossing rate
        audio_array = data.get('audio_data') or data.get('audioArray')
        if not audio_array:
            return
        samples = np.array(audio_array, dtype=np.float32)
        if samples.size == 0:
            return
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        energy = float(np.mean(np.abs(samples)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))))
        # Simple mapping
        scores = {
            'angry': min(1.0, max(0.0, (energy - 0.05) * 8.0 + (zcr - 0.1) * 1.5)),
            'happy': min(1.0, max(0.0, (energy - 0.03) * 5.0 + max(0.0, 0.15 - abs(zcr - 0.15)) * 2.0)),
            'sad':   min(1.0, max(0.0, max(0.0, 0.03 - energy) * 5.0 + max(0.0, 0.1 - zcr) * 3.0)),
            'fearful': min(1.0, max(0.0, (zcr - 0.18) * 3.0)),
            'disgusted': 0.0,
            'surprised': min(1.0, max(0.0, (energy - 0.07) * 10.0)),
            'neutral': 0.0
        }
        total = sum(scores.values())
        if total <= 0:
            scores['neutral'] = 1.0
            total = 1.0
        for k in scores:
            scores[k] = scores[k] / total
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant]
        current_emotions['voice'] = [{
            'emotion': dominant,
            'confidence': float(confidence),
            'source': 'real_audio',
            'audio_energy': energy,
            'all_scores': scores
        }]
        emit('emotion_update', current_emotions)
    except Exception as e:
        logger.error(f"Audio processing error: {e}")


def run_server():
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server()


