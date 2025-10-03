# Voice Shield: AI-Powered Aggression Detector

A comprehensive emotion detection system that combines facial expression analysis and voice emotion recognition to detect aggression levels in real-time.

## Features

- **Real-time Facial Expression Analysis**: Uses face-api.js for detecting 7 different emotions
- **Voice Emotion Detection**: Analyzes audio for voice emotion indicators
- **Combined AI Analysis**: Integrates both visual and audio cues for accurate aggression detection
- **Real-time WebSocket Communication**: Socket.IO for seamless data transmission
- **Modern UI**: Beautiful, responsive interface with real-time visualizations
- **Audio Visualizer**: Real-time audio level visualization
- **Comprehensive Scoring**: Separate scores for facial, voice, and combined analysis

## Technology Stack

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- Tailwind CSS for styling
- Socket.IO client for real-time communication
- face-api.js for facial expression recognition
- Web Audio API for audio processing

### Backend
- Python 3.8+
- Flask web framework
- Flask-SocketIO for WebSocket support
- NumPy for numerical computations
- Audio processing libraries (librosa, pyAudioAnalysis)

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js (optional, for development)
- Modern web browser with camera and microphone support

### Backend Setup

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server**:
   ```bash
   python server.py
   ```
   
   The server will start on `http://localhost:8080`

### Frontend Setup

1. **Open the HTML file**:
   - Simply open `index.html` in a modern web browser
   - Or serve it using a local web server

2. **For development server** (optional):
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Using Node.js
   npx serve .
   ```

## Usage

### Starting the System

1. **Start the backend server**:
   ```bash
   python server.py
   ```

2. **Open the frontend**:
   - Navigate to `index.html` in your browser
   - Or visit `http://localhost:8000` if using a local server

3. **Connect to AI Server**:
   - Click the "Connect to AI Server" button
   - Wait for the connection status to show "Connected"

4. **Start Detection**:
   - Click "Start Detection" to begin camera and microphone access
   - Grant permissions when prompted by your browser

### Understanding the Interface

#### Visual Elements
- **Video Feed**: Real-time camera feed with facial expression overlays
- **Aggression Bar**: Visual indicator of current aggression level
- **Score Cards**: Individual scores for facial, voice, and combined analysis
- **Audio Visualizer**: Real-time audio level visualization
- **Status Indicator**: Connection and recording status

#### Analysis Results
- **Facial Emotion Probabilities**: Real-time breakdown of detected emotions
- **AI Analysis Results**: Combined analysis from the AI server
- **Recommendations**: Action recommendations based on aggression level

### Aggression Levels

- **0-40%**: Calm - No action needed
- **40-60%**: Low Aggression - Continue monitoring
- **60-80%**: Moderate Aggression - Monitor closely
- **80-100%**: High Aggression - Immediate intervention recommended

## API Endpoints

### WebSocket Events

#### Client to Server
- `facial_analysis`: Send facial expression data
- `audio`: Send audio data for analysis
- `get_analysis`: Request current analysis results

#### Server to Client
- `emotion_analysis`: Combined analysis results
- `status`: Server status messages
- `error`: Error messages

### REST API Endpoints

- `GET /health`: Health check endpoint
- `GET /api/analysis`: Get current analysis results
- `GET /`: Serve the main application

## Configuration

### Server Configuration
Edit `server.py` to modify:
- Server port (default: 8080)
- Analysis weights and thresholds
- Audio processing parameters

### Frontend Configuration
Edit `index.html` to modify:
- Server URL (default: `http://localhost:8080`)
- Audio sample rate and chunk size
- UI styling and layout

## Advanced Features

### Custom AI Models
The system can be extended with:
- Custom facial expression models
- Advanced audio emotion detection
- Machine learning models for better accuracy

### Integration Options
- REST API for external systems
- Database storage for historical data
- Alert systems for high aggression levels
- Mobile app integration

## Troubleshooting

### Common Issues

1. **Camera/Microphone Access Denied**
   - Ensure browser permissions are granted
   - Check if other applications are using the camera/microphone

2. **Server Connection Failed**
   - Verify the Flask server is running
   - Check if port 8080 is available
   - Ensure firewall settings allow the connection

3. **Models Not Loading**
   - Check internet connection for CDN resources
   - Verify face-api.js is accessible

4. **Audio Analysis Issues**
   - Ensure microphone is working
   - Check browser audio permissions
   - Verify audio processing libraries are installed

### Performance Optimization

- **Reduce Detection Frequency**: Modify the interval in `startDetection()`
- **Lower Video Quality**: Adjust camera constraints
- **Optimize Audio Processing**: Reduce sample rate or chunk size

## Security Considerations

- **HTTPS**: Use HTTPS in production for secure data transmission
- **Authentication**: Implement user authentication for production use
- **Data Privacy**: Ensure compliance with privacy regulations
- **Input Validation**: Validate all incoming data on the server

## Development

### Project Structure
```
├── index.html          # Frontend application
├── server.py           # Flask backend server
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review browser console for errors
3. Check server logs for backend issues
4. Create an issue with detailed information

## Future Enhancements

- **Cloud AI Integration**: Connect to cloud AI services for better accuracy
- **Mobile App**: Native mobile application
- **Multi-language Support**: Support for multiple languages
- **Advanced Analytics**: Historical data analysis and trends
- **Custom Training**: Ability to train custom models 