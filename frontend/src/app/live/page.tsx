'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';

interface DetectionResult {
  mask: string; // base64 encoded mask
  boundingBoxes: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
  hasDetection: boolean;
}

const LiveDetectionPage: React.FC = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [cameraType, setCameraType] = useState<'webcam' | 'raspberry'>('webcam');
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number>();
  const lastDetectionRef = useRef<DetectionResult | null>(null);
  const frameCountRef = useRef(0);
  const fpsStartTimeRef = useRef(Date.now());
  
  // Same settings as Python script
  const PROCESS_EVERY_N_FRAMES = 5;
  const INFERENCE_SIZE = { width: 640, height: 480 };

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      
      let constraints: MediaStreamConstraints;
      
      if (cameraType === 'webcam') {
        constraints = {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 }
          }
        };
      } else {
        // For Raspberry Pi camera, we'll use a different approach
        // This would typically connect to a Raspberry Pi camera stream
        constraints = {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 }
          }
        };
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Failed to access camera. Please check permissions.');
    }
  }, [cameraType]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    setIsDetecting(false);
    setIsProcessing(false);
    setFps(0);
    setFrameCount(0);
    frameCountRef.current = 0;
    lastDetectionRef.current = null;
  }, []);

  const drawOverlay = useCallback((canvas: HTMLCanvasElement, detection: DetectionResult | null) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !detection || !detection.hasDetection) return;
    
    // Draw red overlay on detected areas (same as Python script)
    if (detection.mask) {
      const maskImg = new Image();
      maskImg.onload = () => {
        // Create a temporary canvas for the red overlay
        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.width = canvas.width;
        overlayCanvas.height = canvas.height;
        const overlayCtx = overlayCanvas.getContext('2d');
        
        if (overlayCtx) {
          // Fill entire canvas with red at 40% opacity (same as Python alpha=0.4)
          overlayCtx.fillStyle = 'rgba(255, 0, 0, 0.4)';
          overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          
          // Apply mask to only show red where detection occurred
          overlayCtx.globalCompositeOperation = 'destination-in';
          overlayCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
          
          // Draw the red overlay on the main canvas (blending with existing frame)
          ctx.globalAlpha = 1.0;
          ctx.globalCompositeOperation = 'source-over';
          ctx.drawImage(overlayCanvas, 0, 0);
        }
        
        // Reset composite operation
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
        
        // Draw bounding boxes (same as Python script)
        detection.boundingBoxes.forEach(box => {
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 2;
          ctx.strokeRect(box.x, box.y, box.width, box.height);
          
          // Draw label
          ctx.fillStyle = '#00FF00';
          ctx.font = 'bold 16px Arial';
          ctx.fillText('Camouflage', box.x, box.y - 10);
        });
      };
      maskImg.src = `data:image/png;base64,${detection.mask}`;
    } else if (detection.boundingBoxes.length > 0) {
      // If we have bounding boxes but no mask, just draw the boxes
      detection.boundingBoxes.forEach(box => {
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        
        ctx.fillStyle = '#00FF00';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('Camouflage', box.x, box.y - 10);
      });
    }
  }, []);

  const processFrame = useCallback(async (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    frameCountRef.current++;
    const currentFrameCount = frameCountRef.current;
    
    // Always draw the current video frame on canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // If we have a last detection result, draw the overlay
    if (lastDetectionRef.current) {
      drawOverlay(canvas, lastDetectionRef.current);
    }
    
    // Only process every Nth frame (same as Python script)
    if (currentFrameCount % PROCESS_EVERY_N_FRAMES !== 0) {
      return;
    }
    
    if (isProcessing) return; // Prevent overlapping requests
    setIsProcessing(true);
    
    try {
      // Create a smaller canvas for inference (same as Python script)
      const inferenceCanvas = document.createElement('canvas');
      const inferenceCtx = inferenceCanvas.getContext('2d');
      if (!inferenceCtx) return;
      
      inferenceCanvas.width = INFERENCE_SIZE.width;
      inferenceCanvas.height = INFERENCE_SIZE.height;
      
      // Draw resized frame for inference
      inferenceCtx.drawImage(video, 0, 0, INFERENCE_SIZE.width, INFERENCE_SIZE.height);
      
      // Convert to blob for API call
      const blob = await new Promise<Blob>((resolve) => {
        inferenceCanvas.toBlob((blob) => {
          resolve(blob!);
        }, 'image/jpeg', 0.8);
      });
      
      // Send to backend for segmentation
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');
      
      const response = await fetch('http://localhost:8000/api/analyze-image', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result: DetectionResult = await response.json();
        lastDetectionRef.current = result;
      } else {
        console.error('Detection failed');
      }
      
    } catch (error) {
      console.error('Frame processing error:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, drawOverlay]);

  const animate = useCallback(() => {
    if (!isDetecting || !videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      processFrame(video, canvas);
      
      // Update FPS
      const now = Date.now();
      const elapsed = (now - fpsStartTimeRef.current) / 1000;
      if (elapsed > 0) {
        setFps(frameCountRef.current / elapsed);
      }
      setFrameCount(frameCountRef.current);
    }
    
    animationRef.current = requestAnimationFrame(animate);
  }, [isDetecting, processFrame]);

  useEffect(() => {
    if (isDetecting) {
      animate();
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isDetecting, animate]);

  const handleStartDetection = async () => {
    await startCamera();
    setIsDetecting(true);
    fpsStartTimeRef.current = Date.now();
    frameCountRef.current = 0;
  };

  const handleStopDetection = () => {
    stopCamera();
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Live Camouflage Detection
          </h1>
          <p className="text-lg text-gray-600">
            Real-time segmentation overlay with the same settings as the Python script
          </p>
        </div>

        {/* Camera Selection */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <span className="mr-2">📷</span>
            Camera Selection
          </h2>
          
          <div className="flex space-x-4 mb-4">
            <button
              onClick={() => setCameraType('webcam')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                cameraType === 'webcam'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Webcam
            </button>
            <button
              onClick={() => setCameraType('raspberry')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                cameraType === 'raspberry'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Raspberry Pi Camera
            </button>
          </div>
          
          <div className="flex space-x-4">
            <button
              onClick={handleStartDetection}
              disabled={isDetecting}
              className="flex items-center px-6 py-3 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <span className="mr-2">▶️</span>
              Start Detection
            </button>
            
            <button
              onClick={handleStopDetection}
              disabled={!isDetecting}
              className="flex items-center px-6 py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <span className="mr-2">⏹️</span>
              Stop Detection
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center">
            <span className="text-red-500 mr-2">⚠️</span>
            <span className="text-red-700">{error}</span>
          </div>
        )}

        {/* Video Display */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <span className="mr-2">📹</span>
            Live Feed
          </h2>
          
          <div className="relative">
            <video
              ref={videoRef}
              className="w-full max-w-4xl mx-auto rounded-lg"
              style={{ display: isDetecting ? 'block' : 'none' }}
              muted
            />
            
            <canvas
              ref={canvasRef}
              className="w-full max-w-4xl mx-auto rounded-lg"
              style={{ display: isDetecting ? 'block' : 'none' }}
            />
            
            {!isDetecting && (
              <div className="w-full max-w-4xl mx-auto h-96 bg-gray-200 rounded-lg flex items-center justify-center">
                <div className="text-center text-gray-500">
                  <span className="text-6xl mb-4 block">📷</span>
                  <p>Camera feed will appear here when detection starts</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Performance Stats */}
        {isDetecting && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <span className="mr-2">⚙️</span>
              Performance Stats
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-medium text-blue-900">FPS</h3>
                <p className="text-2xl font-bold text-blue-600">{fps.toFixed(1)}</p>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-medium text-green-900">Frame Count</h3>
                <p className="text-2xl font-bold text-green-600">{frameCount}</p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="font-medium text-purple-900">Processing</h3>
                <p className="text-2xl font-bold text-purple-600">
                  {isProcessing ? 'Yes' : 'No'}
                </p>
              </div>
            </div>
            
            <div className="mt-4 text-sm text-gray-600">
              <p><strong>Settings:</strong> Processing every {PROCESS_EVERY_N_FRAMES} frames</p>
              <p><strong>Inference Size:</strong> {INFERENCE_SIZE.width}×{INFERENCE_SIZE.height}</p>
              <p><strong>Overlay:</strong> Red mask with green bounding boxes</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveDetectionPage;
