#!/usr/bin/env python3
"""
AI-Powered Aggression Detection Server
Handles real-time audio and facial analysis for emotion detection
"""

import asyncio
import json
import logging
import numpy as np
import websockets
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
from collections import deque
import base64
import os
from flask_cors import CORS
import io
import soundfile as sf
import librosa
import torch
import torchaudio
from transformers import pipeline
from scipy.stats import mode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for storing analysis data
facial_data_buffer = deque(maxlen=10)
audio_data_buffer = deque(maxlen=50)
analysis_results = {}

class EmotionAnalyzer:
    """AI-powered emotion analyzer for combined audio and facial analysis"""
    
    def __init__(self):
        self.facial_weights = {
            'angry': 0.7,
            'fearful': 0.2,
            'disgusted': 0.1
        }
        
        # Voice emotion classifier using wav2vec2 model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.voice_classifier = pipeline(
                task="audio-classification",
                model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                device=device,
                top_k=None  # Return all scores
            )
            logger.info(f"Loaded voice emotion model on {device}")
        except Exception as e:
            logger.error(f"Failed to load voice emotion model: {e}")
            self.voice_classifier = None
            
        # Initialize confidence thresholds
        self.confidence_thresholds = {
            'angry': 0.45,    # Lower threshold for better sensitivity
            'happy': 0.40,
            'sad': 0.40,
            'fearful': 0.45,
            'disgusted': 0.45,
            'surprised': 0.40,
            'neutral': 0.35
        }
        
        # Initialize emotion history
        self.emotion_history = {
            'angry': deque(maxlen=5),
            'happy': deque(maxlen=5),
            'sad': deque(maxlen=5),
            'fearful': deque(maxlen=5),
            'disgusted': deque(maxlen=5),
            'surprised': deque(maxlen=5),
            'neutral': deque(maxlen=5)
        }

        # Initialize voice detection parameters
        self.vad_energy_threshold = 0.01
        self.vad_duration_threshold = 0.2

    def analyze_facial_emotions(self, expressions):
        """Analyze facial expressions for aggression indicators"""
        if not expressions:
            return {'aggression_score': 0, 'confidence': 0, 'emotions': {}}
        
        aggression_score = 0
        for emotion, weight in self.facial_weights.items():
            if emotion in expressions:
                aggression_score += expressions[emotion] * weight * 100
        
        if aggression_score < 10 and (
            expressions.get('angry', 0) > 0.005 or
            expressions.get('fearful', 0) > 0.005 or
            expressions.get('disgusted', 0) > 0.005
        ):
            aggression_score = 10
        
        max_expression = max(expressions.values()) if expressions else 0
        confidence = min(max_expression * 100, 95)
        
        # Convert to percentages and round to one decimal place
        emotion_bars = {k: round(v * 100, 1) for k, v in expressions.items()}
        
        return {
            'aggression_score': round(min(aggression_score, 100), 1),
            'confidence': round(confidence, 1),
            'emotions': {k: round(v * 100, 1) for k, v in expressions.items()},
            'emotion_bars': emotion_bars
        }

    def analyze_audio_emotions(self, audio_data_b64: str):
        """Analyze 16kHz mono WAV base64 string and return voice emotion scores."""
        if not audio_data_b64 or not self.voice_classifier:
            return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}
        try:
            wav_bytes = base64.b64decode(audio_data_b64)
            audio_io = io.BytesIO(wav_bytes)
            samples, sr = sf.read(audio_io, dtype='float32', always_2d=False)
            
            # Convert to mono if stereo
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
                sr = 16000
                
            # Normalize audio
            samples = librosa.util.normalize(samples)
            
            # Enhanced preprocessing
            # 1. Noise reduction using spectral gating
            samples = librosa.decompose.nn_filter(samples,
                                                aggregate=np.median,
                                                metric='cosine',
                                                width=3)
            
            # 2. Voice activity detection
            rms = np.sqrt(np.mean(np.square(samples)))
            duration = len(samples) / sr
            
            if rms < self.vad_energy_threshold or duration < self.vad_duration_threshold:
                return {'voice_emotion': 'neutral',
                       'aggression_score': 0.0,
                       'confidence': 95.0,
                       'audio_bars': {e: (100.0 if e == 'neutral' else 0.0) for e in 
                                    ['angry', 'happy', 'sad', 'fearful', 'disgusted', 'surprised', 'neutral']},
                       'multi_voice': False,
                       'no_input': True}

            # 3. Feature extraction
            mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=20)
            spectral_contrast = librosa.feature.spectral_contrast(y=samples, sr=sr)
            
            # Run classifier
            result = self.voice_classifier({
                'array': samples,
                'sampling_rate': sr
            }, top_k=None)
            
            # Initialize target emotions
            target_emotions = ['angry', 'happy', 'sad', 'fearful', 'disgusted', 'surprised', 'neutral']
            scores = {e: 0.0 for e in target_emotions}
            
            # Process classifier results with improved mapping
            raw_scores = {}
            for item in result:
                label = item['label'].lower()
                score = float(item['score'])
                
                # Enhanced emotion mapping
                if 'ang' in label or 'mad' in label or 'rage' in label:
                    target_label = 'angry'
                elif 'hap' in label or 'joy' in label or 'excit' in label:
                    target_label = 'happy'
                elif 'sad' in label or 'depress' in label or 'unhap' in label:
                    target_label = 'sad'
                elif 'fear' in label or 'nerv' in label or 'worr' in label:
                    target_label = 'fearful'
                elif 'disgust' in label or 'unpleas' in label:
                    target_label = 'disgusted'
                elif 'surpr' in label or 'amaz' in label:
                    target_label = 'surprised'
                elif 'neut' in label or 'calm' in label:
                    target_label = 'neutral'
                else:
                    continue
                
                if target_label in scores:
                    raw_scores[target_label] = raw_scores.get(target_label, 0) + score
            
            # Normalize raw scores
            total_score = sum(raw_scores.values()) if raw_scores else 1
            for emotion in scores:
                scores[emotion] = raw_scores.get(emotion, 0) / total_score
            
            # Enhanced emotion classification using acoustic features
            energy = librosa.feature.rms(y=samples)[0].mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=samples)[0].mean()
            
            # Adjust scores based on acoustic features
            if energy > 0.2 and zero_crossing_rate > 0.15:  # High energy and variability
                if scores['angry'] > 0.3:
                    scores['angry'] *= 1.2  # Boost anger detection
                if scores['happy'] > 0.3:
                    scores['happy'] *= 1.1  # Boost happiness detection
            elif energy < 0.1 and zero_crossing_rate < 0.1:  # Low energy and variability
                if scores['sad'] > 0.3:
                    scores['sad'] *= 1.2  # Boost sadness detection
                if scores['fearful'] > 0.3:
                    scores['fearful'] *= 1.1  # Boost fear detection
            
            # Normalize after adjustments
            total = sum(scores.values())
            if total > 0:
                for emotion in scores:
                    scores[emotion] = scores[emotion] / total
            
            # Apply temporal smoothing with faster response
            for emotion in scores:
                self.emotion_history[emotion].append(scores[emotion])
                recent_scores = list(self.emotion_history[emotion])[-3:]
                scores[emotion] = np.mean(recent_scores)
            
            # Convert to percentages
            audio_bars = {k: round(v * 100, 1) for k, v in scores.items()}
            
            # Get top emotion and confidence
            sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_emotion = sorted_emotions[0][0]
            top_confidence = sorted_emotions[0][1] * 100
            
            # Calculate aggression score
            aggression_score = (
                scores['angry'] * 70 +
                scores['fearful'] * 20 +
                scores['disgusted'] * 10
            )
            
            return {
                'voice_emotion': top_emotion,
                'aggression_score': round(min(aggression_score * 100, 100), 1),
                'confidence': round(min(top_confidence, 95), 1),
                'audio_bars': audio_bars,
                'multi_voice': False
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}", exc_info=True)
            return {
                'voice_emotion': 'Unknown',
                'aggression_score': 0,
                'confidence': 0,
                'audio_bars': {},
                'multi_voice': False
            }

    def combine_analysis(self, facial_result, audio_result):
        """Combine facial and audio analysis for comprehensive aggression detection"""
        facial_weight = 0.6
        audio_weight = 0.4
        
        facial_score = facial_result.get('aggression_score', 0)
        audio_score = audio_result.get('aggression_score', 0)
        combined_score = (facial_score * facial_weight) + (audio_score * audio_weight)
        
        facial_confidence = facial_result.get('confidence', 0)
        audio_confidence = audio_result.get('confidence', 0)
        combined_confidence = (facial_confidence * facial_weight) + (audio_confidence * audio_weight)
        
        if combined_score > 80:
            recommendation = "High risk - Immediate intervention recommended"
        elif combined_score > 60:
            recommendation = "Moderate risk - Monitor closely"
        elif combined_score > 40:
            recommendation = "Low risk - Continue monitoring"
        else:
            recommendation = "No action needed"
        
        return {
            'combined_score': round(combined_score, 1),
            'combined_confidence': round(combined_confidence, 1),
            'facial_score': round(facial_score, 1),
            'audio_score': round(audio_score, 1),
            'voice_emotion': audio_result.get('voice_emotion', 'Unknown'),
            'recommendation': recommendation
        }

# Initialize the emotion analyzer
emotion_analyzer = EmotionAnalyzer()

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('status', {'message': 'Connected to AI server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('facial_analysis')
def handle_facial_analysis(data):
    """Handle incoming facial analysis data"""
    try:
        expressions = data.get('expressions', {})
        face_index = data.get('faceIndex', 0)
        # Prefer a stable faceId from the client when available to avoid cross-mapping
        face_id = data.get('faceId', face_index)
        
        facial_data_buffer.append({
            'expressions': expressions,
            'timestamp': time.time(),
            'face_index': face_index,
            'face_id': face_id
        })
        
        current_time = time.time()
        facial_data_buffer_list = list(facial_data_buffer)
        facial_data_buffer.clear()
        facial_data_buffer.extend([entry for entry in facial_data_buffer_list if current_time - entry['timestamp'] < 5.0])
        
        facial_result = emotion_analyzer.analyze_facial_emotions(expressions)
        # Store per-face result keyed by stable face_id
        analysis_results[f'facial_face_{face_id}'] = facial_result
        
        # Get facial results sorted by face index
        facial_results = [(k, v) for k, v in analysis_results.items() 
                          if k.startswith('facial_face_') and isinstance(v, dict)]
        # Order does not matter for aggregation; keep as-is to preserve mapping stability
        all_facial_results = [v for _, v in facial_results]
        
        if all_facial_results:
            avg_aggression_score = sum(r.get('aggression_score', 0) for r in all_facial_results) / len(all_facial_results)
            avg_confidence = sum(r.get('confidence', 0) for r in all_facial_results) / len(all_facial_results)
            emotion_keys = set()
            for r in all_facial_results:
                emotion_keys.update(r.get('emotion_bars', {}).keys())
            avg_emotion_bars = {k: round(sum(r.get('emotion_bars', {}).get(k, 0) for r in all_facial_results) / len(all_facial_results), 1) for k in emotion_keys}
            aggregated_facial_result = {
                'aggression_score': round(avg_aggression_score, 1),
                'confidence': round(avg_confidence, 1),
                'emotions': {k: round(v * 100, 1) for k, v in expressions.items()},
                'emotion_bars': {ek: round(float(ev), 1) for ek, ev in avg_emotion_bars.items()},
                'face_count': len(all_facial_results)
            }
            analysis_results['facial'] = aggregated_facial_result
        else:
            aggregated_facial_result = facial_result
        
        if 'audio' in analysis_results and analysis_results.get('audio'):
            combined_result = emotion_analyzer.combine_analysis(
                aggregated_facial_result,
                analysis_results['audio']
            )
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': combined_result['voice_emotion'],
                'combinedScore': round(float(combined_result['combined_score']), 1),
                'confidence': round(float(combined_result['combined_confidence']), 1),
                'recommendation': combined_result['recommendation'],
                'facialScore': round(float(combined_result['facial_score']), 1),
                'audioScore': round(float(combined_result['audio_score']), 1),
                'faceCount': aggregated_facial_result.get('face_count', 1),
                'emotionBars': {k: round(float(v), 1) for k, v in aggregated_facial_result.get('emotion_bars', {}).items()},
                'audioBars': {k: round(float(v), 1) for k, v in analysis_results['audio'].get('audio_bars', {}).items()},
                'multiVoice': int(analysis_results['audio'].get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': 'Analyzing...',
                'combinedScore': round(float(aggregated_facial_result.get('aggression_score', 0)), 1),
                'confidence': round(float(aggregated_facial_result.get('confidence', 0)), 1),
                'recommendation': f'Monitoring {aggregated_facial_result.get("face_count", 1)} face(s)...',
                'facialScore': round(float(aggregated_facial_result.get('aggression_score', 0)), 1),
                'audioScore': 0.0,
                'faceCount': aggregated_facial_result.get('face_count', 1),
                'emotionBars': {k: round(float(v), 1) for k, v in aggregated_facial_result.get('emotion_bars', {}).items()}
            })

    except Exception as e:
        logger.error(f"Error processing facial analysis: {e}", exc_info=True)
        emit('error', {'message': 'Error processing facial analysis'})

@socketio.on('audio')
def handle_audio_data(data):
    """Handle incoming audio data"""
    try:
        # If the client indicates no input, emit neutral quickly and return
        if data.get('no_input'):
            analysis_results['audio'] = {
                'voice_emotion': 'neutral',
                'aggression_score': 0.0,
                'confidence': 0.0,
                'audio_bars': {e: (100.0 if e == 'neutral' else 0.0) for e in ['angry','happy','sad','fearful','disgusted','surprised','neutral']},
                'multi_voice': False,
                'no_input': True
            }
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': 'neutral',
                'combinedScore': round(float(analysis_results.get('facial', {}).get('aggression_score', 0)), 1),
                'confidence': 0.0,
                'recommendation': 'No voice input',
                'facialScore': round(float(analysis_results.get('facial', {}).get('aggression_score', 0)), 1),
                'audioScore': 0.0,
                'faceCount': analysis_results.get('facial', {}).get('face_count', 0),
                'emotionBars': analysis_results.get('facial', {}).get('emotion_bars', {}),
                'audioBars': analysis_results['audio'].get('audio_bars', {}),
                'multiVoice': 0
            })
            return

        audio_data = data.get('data')
        audio_data_buffer.append({
            'data': audio_data,
            'timestamp': time.time()
        })
        
        audio_result = emotion_analyzer.analyze_audio_emotions(audio_data)
        analysis_results['audio'] = audio_result

        if 'facial' in analysis_results and analysis_results.get('facial'):
            combined_result = emotion_analyzer.combine_analysis(
                analysis_results['facial'],
                audio_result
            )
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': combined_result['voice_emotion'],
                'combinedScore': round(float(combined_result['combined_score']), 1),
                'confidence': round(float(combined_result['combined_confidence']), 1),
                'recommendation': combined_result['recommendation'],
                'facialScore': round(float(combined_result['facial_score']), 1),
                'audioScore': round(float(combined_result['audio_score']), 1),
                'faceCount': analysis_results['facial'].get('face_count', 1),
                'emotionBars': {k: round(float(v), 1) for k, v in analysis_results['facial'].get('emotion_bars', {}).items()},
                'audioBars': {k: round(float(v), 1) for k, v in audio_result.get('audio_bars', {}).items()},
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': audio_result.get('voice_emotion', 'Unknown'),
                'combinedScore': round(float(audio_result.get('aggression_score', 0)), 1),
                'confidence': round(float(audio_result.get('confidence', 0)), 1),
                'recommendation': 'Monitoring voice patterns...',
                'facialScore': 0.0,
                'audioScore': round(float(audio_result.get('aggression_score', 0)), 1),
                'faceCount': 0,
                'emotionBars': {},
                'audioBars': {k: round(float(v), 1) for k, v in audio_result.get('audio_bars', {}).items()},
                'multiVoice': int(audio_result.get('multi_voice', False))
            })

    except Exception as e:
        logger.error(f"Error processing audio data: {e}", exc_info=True)
        emit('error', {'message': 'Error processing audio data'})

@app.route('/')
def index():
    return render_template('index.html')

def run_server():
    """Run the Flask server with SocketIO"""
    logger.info("Starting AI Aggression Detection Server...")
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting at http://localhost:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    run_server()