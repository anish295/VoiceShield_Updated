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

# Remove all SpeechBrain and related model loading code

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

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
        # No model loading required
        
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
        emotion_bars = {k: round(v * 100, 2) for k, v in expressions.items()}
        
        return {
            'aggression_score': min(aggression_score, 100),
            'confidence': confidence,
            'emotions': expressions,
            'emotion_bars': emotion_bars
        }

    def analyze_audio_emotions(self, audio_data):
        """Set voice emotion output to match the current facial emotion output (averaged across all faces)."""
        if not audio_data:
            return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}
        try:
            # Decode base64 audio to bytes
            audio_bytes = base64.b64decode(audio_data)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_np) == 0:
                return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}

            # If facial result exists, use it for voice output
            facial_result = analysis_results.get('facial')
            if facial_result and 'emotion_bars' in facial_result:
                emotion_bars = facial_result['emotion_bars']
                voice_emotion = max(emotion_bars, key=emotion_bars.get)
                aggression_score = int(emotion_bars.get("angry", 0) * 0.7 + emotion_bars.get("fearful", 0) * 0.2 + emotion_bars.get("disgusted", 0) * 0.1)
                confidence = int(max(emotion_bars.values()))
                return {
                    'voice_emotion': voice_emotion,
                    'aggression_score': aggression_score,
                    'confidence': confidence,
                    'audio_bars': emotion_bars,
                    'multi_voice': False
                }
            # Fallback: neutral
            emotions = ["angry", "happy", "sad", "fearful", "disgusted", "surprised", "neutral"]
            audio_bars = {e: (100 if e == "neutral" else 0) for e in emotions}
            return {
                'voice_emotion': "neutral",
                'aggression_score': 0,
                'confidence': 100,
                'audio_bars': audio_bars,
                'multi_voice': False
            }
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}
    
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
            'facial_score': facial_score,
            'audio_score': audio_score,
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
        # ... (rest of the function is unchanged)
        face_index = data.get('faceIndex', 0)
        
        facial_data_buffer.append({
            'expressions': expressions,
            'timestamp': time.time(),
            'face_index': face_index
        })
        
        current_time = time.time()
        facial_data_buffer_list = list(facial_data_buffer)
        facial_data_buffer.clear()
        facial_data_buffer.extend([entry for entry in facial_data_buffer_list if current_time - entry['timestamp'] < 5.0])
        
        facial_result = emotion_analyzer.analyze_facial_emotions(expressions)
        analysis_results[f'facial_face_{face_index}'] = facial_result
        
        all_facial_results = [v for k, v in analysis_results.items() if k.startswith('facial_face_') and isinstance(v, dict)]
        if all_facial_results:
            avg_aggression_score = sum(r.get('aggression_score', 0) for r in all_facial_results) / len(all_facial_results)
            avg_confidence = sum(r.get('confidence', 0) for r in all_facial_results) / len(all_facial_results)
            emotion_keys = set()
            for r in all_facial_results:
                emotion_keys.update(r.get('emotion_bars', {}).keys())
            avg_emotion_bars = {k: round(sum(r.get('emotion_bars', {}).get(k, 0) for r in all_facial_results) / len(all_facial_results), 2) for k in emotion_keys}
            aggregated_facial_result = {
                'aggression_score': avg_aggression_score,
                'confidence': avg_confidence,
                'emotions': expressions,
                'emotion_bars': avg_emotion_bars,
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
                'combinedScore': combined_result['combined_score'],
                'confidence': combined_result['combined_confidence'],
                'recommendation': combined_result['recommendation'],
                'facialScore': combined_result['facial_score'],
                'audioScore': combined_result['audio_score'],
                'faceCount': aggregated_facial_result.get('face_count', 1),
                'emotionBars': aggregated_facial_result.get('emotion_bars', {}),
                'audioBars': analysis_results['audio'].get('audio_bars', {}),
                'multiVoice': int(analysis_results['audio'].get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': 'Analyzing...',
                'combinedScore': aggregated_facial_result.get('aggression_score', 0),
                'confidence': aggregated_facial_result.get('confidence', 0),
                'recommendation': f'Monitoring {aggregated_facial_result.get("face_count", 1)} face(s)...',
                'facialScore': aggregated_facial_result.get('aggression_score', 0),
                'audioScore': 0,
                'faceCount': aggregated_facial_result.get('face_count', 1),
                'emotionBars': aggregated_facial_result.get('emotion_bars', {})
            })

    except Exception as e:
        logger.error(f"Error processing facial analysis: {e}", exc_info=True)
        emit('error', {'message': 'Error processing facial analysis'})

@socketio.on('audio')
def handle_audio_data(data):
    """Handle incoming audio data"""
    try:
        audio_data = data.get('data')
        # ... (rest of the function is unchanged)
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
                'combinedScore': combined_result['combined_score'],
                'confidence': combined_result['combined_confidence'],
                'recommendation': combined_result['recommendation'],
                'facialScore': combined_result['facial_score'],
                'audioScore': combined_result['audio_score'],
                'faceCount': analysis_results['facial'].get('face_count', 1),
                'emotionBars': analysis_results['facial'].get('emotion_bars', {}),
                'audioBars': audio_result.get('audio_bars', {}),
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': audio_result.get('voice_emotion', 'Unknown'),
                'combinedScore': audio_result.get('aggression_score', 0),
                'confidence': audio_result.get('confidence', 0),
                'recommendation': 'Monitoring voice patterns...',
                'facialScore': 0,
                'audioScore': audio_result.get('aggression_score', 0),
                'faceCount': 0,
                'emotionBars': {},
                'audioBars': audio_result.get('audio_bars', {}),
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
    except Exception as e:
        logger.error(f"Error processing audio data: {e}", exc_info=True)
        emit('error', {'message': 'Error processing audio data'})


@app.route('/')
def index():
    # This route is now optional if you open index.html directly
    return render_template('index.html')

def run_server():
    """Run the Flask server with SocketIO"""
    logger.info("Starting AI Aggression Detection Server...")
    socketio.run(app, host='0.0.0.0', port=8081, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    run_server()