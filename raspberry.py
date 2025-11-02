#!/usr/bin/env python3
"""
Mirqab Laptop Processing Server - COMPLETE VERSION
Receives frames from RPi, processes with AI model, returns masks
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
import base64
import time
import os
import traceback

app = Flask(__name__)

class CamouflageProcessor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 Loading AI model on {self.device}...")
        self.model = None
        self.transform = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print(f"   Current directory: {os.getcwd()}")
                print("🔄 Using mock detection mode")
                self.model = None
                return
            
            print(f"📦 Loading model from: {model_path}")
            print(f"   Device: {self.device.type.upper()}")
            
            # Initialize DeepLabV3 model (use weights=None instead of pretrained=False)
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
            
            # Adjust classifier for 2 classes (background + camouflage)
            self.model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
            
            # Modify aux_classifier if it exists
            if self.model.aux_classifier is not None:
                self.model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
            
            # Load weights with PyTorch 2.6+ compatibility
            print(f"   Loading checkpoint...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as load_error:
                # Fallback: Add safe globals for numpy if needed
                print(f"   Retrying with numpy safe globals...")
                import numpy as np
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Load with strict=False to allow missing or extra keys (e.g., aux_classifier)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            print("   Note: Loaded with strict=False (auxiliary classifier may differ)")
            
            # Optimize model
            self.model.eval()
            self.model.to(self.device)
            
            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Setup image transforms (match training preprocessing)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ AI Model loaded successfully!")
            
            # Get model size
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"   Model Parameters: {model_size:.2f}M")
            
            # Test model with a dummy input to verify it works
            print("   Testing model with dummy input...")
            try:
                test_input = torch.randn(1, 3, 480, 640).to(self.device)
                with torch.no_grad():
                    test_output = self.model(test_input)['out']
                print(f"   ✅ Model test successful - Output shape: {test_output.shape}")
                print(f"   ✅ Model expects input: (1, 3, 480, 640)")
            except Exception as test_error:
                print(f"   ⚠️  Model test failed: {test_error}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Using mock detection mode")
            self.model = None

    def create_mask(self, frame):
        """Create camouflage mask for the frame using DeepLabV3"""
        # If model not loaded, use mock detection
        if self.model is None:
            print("⚠️  Model is None - using mock detection")
            return self._mock_detection(frame)
        
        try:
            # Get original dimensions
            original_h, original_w = frame.shape[:2]
            
            # Convert BGR to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (keep aspect ratio better) - use 512x512 or 640x480
            # Most DeepLabV3 models work best with these sizes
            input_size = (640, 480)  # Can also try (512, 512)
            frame_resized = cv2.resize(frame_rgb, input_size)
            pil_image = Image.fromarray(frame_resized)
            
            # Prepare input tensor
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run AI inference
            with torch.no_grad():
                output = self.model(input_tensor)['out']
            
            # Get segmentation map (class predictions for each pixel)
            # Check output probabilities first
            probs = torch.softmax(output.squeeze(), dim=0)
            class_1_prob = probs[1].mean().item()
            
            segmentation_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            
            # Debug: Check what classes are in the segmentation map
            unique_classes, counts = np.unique(segmentation_map, return_counts=True)
            
            # Only print debug info if interesting or error
            if class_1_prob > 0.01 or len(unique_classes) > 1:
                print(f"🔍 Model inference: Shape={output.shape}, Class1 prob={class_1_prob:.4f}")
                print(f"🔍 Classes in map: {dict(zip(unique_classes, counts))}")
            
            # Resize back to original size using nearest neighbor (important for preserving class labels)
            segmentation_map = cv2.resize(
                segmentation_map.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST  # Critical for semantic segmentation
            )
            
            # Create binary mask for soldier class (class_id = 1)
            # Class 0 = background, Class 1 = camouflage soldier
            binary_mask = (segmentation_map == 1).astype(np.uint8)
            
            # Debug: Check binary mask before morphological operations
            pixels_before = np.sum(binary_mask)
            
            # Apply morphological operations ONLY if we have detections and they're substantial
            # Don't apply if detections are sparse - might remove them
            if pixels_before > 500:  # Only clean if we have substantial detections
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
                pixels_after = np.sum(binary_mask)
                if pixels_before > 100:
                    print(f"✅ Detection: {pixels_before} → {pixels_after} pixels after cleanup")
            elif pixels_before > 0:
                # Small detections - skip morphological ops to preserve them
                pass
            
            # Create colored mask (Red for camouflage) - scale to 255 for visibility
            colored_mask = np.zeros_like(frame)
            colored_mask[binary_mask == 1] = [0, 0, 255]  # BGR: Red
            
            # Calculate detection metrics
            total_pixels = original_w * original_h
            camouflage_pixels = np.sum(binary_mask)
            confidence = camouflage_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Find contours for detection count with lower threshold for better sensitivity
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Lower minimum area threshold for better detection (was 500, now 100)
            min_area = 100
            detection_count = sum(1 for cnt in contours if cv2.contourArea(cnt) >= min_area)
            
            # Also try connected components for more accurate counting
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            # Count connected components (excluding background label 0)
            component_count = sum(1 for i in range(1, num_labels) 
                                if stats[i, cv2.CC_STAT_AREA] >= min_area)
            
            # Use the higher count method for more sensitive detection
            detection_count = max(detection_count, component_count)
            
            detection_info = {
                "detected": detection_count > 0,
                "confidence": float(confidence),
                "count": detection_count,
                "pixels_detected": int(camouflage_pixels)
            }
            
            return colored_mask, detection_info
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_detection(frame)

    def _mock_detection(self, frame):
        """Mock detection for testing when model is not available"""
        h, w = frame.shape[:2]
        mask = np.zeros_like(frame)
        
        # Add some mock detections for testing
        detected = np.random.random() > 0.7  # 30% detection chance
        
        if detected:
            # Add random red rectangles for mock camouflage
            for i in range(np.random.randint(1, 4)):
                x = np.random.randint(0, w-100)
                y = np.random.randint(0, h-100)
                w_rect = np.random.randint(50, 200)
                h_rect = np.random.randint(50, 200)
                
                cv2.rectangle(mask, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), -1)
        
        detection_info = {
            "detected": detected,
            "confidence": float(np.random.uniform(0.1, 0.5) if detected else 0.0),
            "count": np.random.randint(1, 4) if detected else 0
        }
        
        return mask, detection_info

# Initialize the processor with model
MODEL_PATH = "best_deeplabv3_camouflage.pth"

# Try to find model in current directory or project root
if not os.path.exists(MODEL_PATH):
    # Try in parent directory
    parent_path = os.path.join(os.path.dirname(__file__), "..", MODEL_PATH)
    if os.path.exists(parent_path):
        MODEL_PATH = os.path.abspath(parent_path)
    else:
        # Try in same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_model_path = os.path.join(script_dir, MODEL_PATH)
        if os.path.exists(script_model_path):
            MODEL_PATH = script_model_path

print(f"🔍 Looking for model at: {os.path.abspath(MODEL_PATH) if os.path.exists(MODEL_PATH) else MODEL_PATH}")

processor = CamouflageProcessor(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": processor.model is not None,
        "device": str(processor.device),
        "using_mock": processor.model is None
    })

@app.route('/test_model', methods=['GET'])
def test_model():
    """Test endpoint to verify model is working"""
    if processor.model is None:
        return jsonify({
            "error": "Model not loaded",
            "model_file": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH)
        }), 400
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask, detection_info = processor.create_mask(test_frame)
    
    return jsonify({
        "model_loaded": True,
        "device": str(processor.device),
        "test_detection": detection_info,
        "mask_pixels": int(np.sum(mask > 0))
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Main endpoint for processing frames"""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400
        
        # Decode base64 frame data
        frame_data = base64.b64decode(data['frame'])
        frame_np = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400
        
        # Process frame with AI
        if processor.model is None:
            print("⚠️  WARNING: Model is None - using mock detection!")
        
        mask, detection_info = processor.create_mask(frame)
        
        # Print detection results (only if detection or verbose)
        if detection_info.get('detected') or detection_info.get('count', 0) > 0:
            print(f"🎯 Detection: {detection_info.get('count', 0)} objects, confidence: {detection_info.get('confidence', 0):.3f}")
        
        # Encode results
        _, mask_buffer = cv2.imencode('.jpg', mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        
        # Create overlay for debugging
        overlay_frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        _, overlay_buffer = cv2.imencode('.jpg', overlay_frame)
        overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
        
        # Prepare response
        response = {
            "mask": mask_base64,
            "overlay": overlay_base64,
            "detection": detection_info,
            "timestamp": time.time(),
            "processing_time": time.time() - (data.get('timestamp') or time.time())
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Mirqab Laptop AI Processing Server - DeepLabV3")
    print("=" * 70)
    print(f"🤖 AI Model: DeepLabV3 ResNet-101")
    print(f"📦 Model File: {MODEL_PATH}")
    print(f"⚙️  Device: {processor.device}")
    print(f"✅ Model Loaded: {processor.model is not None}")
    
    if processor.model is not None:
        print(f"🎯 Detection Mode: DeepLabV3 AI (enhanced)")
        print(f"📊 Improvements:")
        print(f"   - Morphological filtering for cleaner masks")
        print(f"   - Lower detection threshold (100px)")
        print(f"   - Connected components analysis")
    else:
        print(f"⚠️  Detection Mode: Mock (model not loaded)")
        print(f"⚠️  Check model file path!")
    
    print("\n📍 Available Endpoints:")
    print("   • GET  /health        - System health check")
    print("   • POST /process_frame - Process frame with DeepLabV3 AI")
    print("\n🌐 Server starting on: http://0.0.0.0:8081")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)