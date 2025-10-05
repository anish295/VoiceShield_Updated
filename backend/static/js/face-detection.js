// Face detection and expression analysis script
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('videoFeed');
  const canvas = document.getElementById('faceCanvas');
  const ctx = canvas.getContext('2d');
  let isProcessing = false;
  
  // Fix for the facial expression box direction issue
  // This corrects the mirroring effect when drawing detection boxes
  async function setupFaceDetection() {
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri('/static/models');
      await faceapi.nets.faceExpressionNet.loadFromUri('/static/models');
      
      video.addEventListener('play', () => {
        const displaySize = { width: video.width, height: video.height };
        faceapi.matchDimensions(canvas, displaySize);
        
        setInterval(async () => {
          if (isProcessing || !video.paused && !video.ended) {
            isProcessing = true;
            
            // Detect faces with expressions
            const detections = await faceapi.detectAllFaces(
              video, 
              new faceapi.TinyFaceDetectorOptions()
            ).withFaceExpressions();
            
            // Clear previous drawings
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Fix direction issue by flipping the context horizontally
            ctx.save();
            ctx.scale(-1, 1); // This flips horizontally
            ctx.translate(-canvas.width, 0);
            
            // Draw the detections with corrected coordinates
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            faceapi.draw.drawDetections(canvas, resizedDetections);
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
            
            // Restore the context to normal
            ctx.restore();
            
            isProcessing = false;
          }
        }, 100);
      });
    } catch (error) {
      console.error('Error setting up face detection:', error);
    }
  }
  
  // Initialize face detection when video is ready
  video.addEventListener('loadedmetadata', setupFaceDetection);
});