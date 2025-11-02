"""
Real-time Segmentation with PixelLib Integration
Uses custom PyTorch DeepLabV3 model: best_deeplabv3_camouflage.pth
Integrated with Firebase for report storage and image backup
GPU OPTIMIZED VERSION with AUTO-REPORTING

=================================================================
FEATURES:
=================================================================
✅ Real-time camouflage detection from webcam
✅ GPU acceleration with CUDA support
✅ Mixed precision (FP16) for faster inference
✅ Automatic frame capture when camouflage detected
✅ Automatic report generation every 2 minutes (prevents spam)
✅ AI analysis using OpenAI GPT-4 Vision API
✅ Image upload to Firebase Storage
✅ Report generation and storage in Firestore database
✅ Visual countdown timer for next auto-report
✅ Local frame backup with timestamps

=================================================================
USAGE:
=================================================================
1. Start the script:
   python segmentation.py

2. Keyboard controls during real-time stream:
   - 'q'  : Quit the application
   - 's'  : Save current frame locally
   - 'g'  : Manually generate report (respects 2-minute cooldown)

3. Auto-reporting behavior:
   - When camouflage is detected, reports are automatically generated
   - Reports are sent to Firebase every 2 minutes (no spam)
   - Countdown timer shows time until next report
   - Manual 'g' key still works but respects the cooldown timer

4. Report includes:
   - Timestamp and location data
   - Environment analysis
   - Number of soldiers detected
   - Attire and camouflage description
   - Equipment details
   - Uploaded snapshot image

=================================================================
REQUIREMENTS:
=================================================================
1. Firebase Setup:
   - FIREBASE_CREDENTIALS_PATH environment variable set
   - Point to your Firebase service account JSON file
   
2. OpenAI API Setup:
   - OPENAI_API_KEY environment variable set
   - Get your API key from https://platform.openai.com

3. Python Packages:
   - cv2 (OpenCV)
   - torch & torchvision (with CUDA support for GPU)
   - PIL
   - firebase-admin
   - openai
   - python-dotenv

=================================================================
ENVIRONMENT VARIABLES:
=================================================================
Set these in your .env file or system environment:

FIREBASE_CREDENTIALS_PATH=/path/to/firebase-adminsdk.json
OPENAI_API_KEY=your-openai-api-key

Optional GPU settings:
USE_GPU=true              # Force GPU usage
GPU_DEVICE=0              # GPU device ID (0 for first GPU)
USE_MIXED_PRECISION=true  # Use FP16 for faster inference
ENABLE_CUDNN_BENCHMARK=true  # Enable cuDNN benchmarking
"""

import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import base64
import io
import asyncio
import traceback
import time
import json
import requests
import subprocess
import platform
from datetime import datetime
from typing import Optional, Tuple, Dict
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Firebase and backend integration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import Firebase handlers
from backend.firestore_handler import FirestoreHandler
from backend.firebase_storage_handler import FirebaseStorageHandler
from backend.llm_handler import LLMReportGenerator

# ============================================================================
# GPU Initialization and Optimization
# ============================================================================

def init_gpu():
    """Initialize GPU and return device"""
    print("\n" + "=" * 70)
    print("🖥️  GPU Initialization")
    print("=" * 70)
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    # Display PyTorch build info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Version: {torch.version.cuda if torch.version.cuda else 'Not available (CPU-only build)'}")
    
    if cuda_available:
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        
        # Optimize cuDNN
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN Benchmarking: Enabled")
        
        device = torch.device("cuda:0")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available or disabled, using CPU (slower)")
        print("\n💡 To enable GPU acceleration:")
        print("   1. Check if you have an NVIDIA GPU:")
        print("      Run: nvidia-smi")
        print("   2. Install CUDA-enabled PyTorch:")
        print("      Visit: https://pytorch.org/get-started/locally/")
        print("      Example (CUDA 11.8):")
        print("      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Or for CUDA 12.1:")
        print("      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("=" * 70)
    return device

# ============================================================================
# Custom Semantic Segmentation with PyTorch Model
# ============================================================================

class CustomSemanticSegmentation:
    """
    Custom semantic segmentation class that mimics PixelLib interface
    but uses PyTorch models with GPU optimization
    """
    
    def __init__(self, use_mixed_precision=True):
        self.model = None
        self.transform = None
        self.device = None
        self.input_size = (640, 480)
        self.use_mixed_precision = use_mixed_precision
        self.scaler = None
        self.fps_counter = 0
        self.last_time = None
        
    def load_model(self, model_path):
        """Load custom DeepLabV3 model with GPU optimization"""
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        try:
            # Initialize device
            self.device = init_gpu()
            print(f"\n📦 Loading model from: {model_path}")
            print(f"   Device: {self.device.type.upper()}")
            print(f"   Mixed Precision (FP16): {self.use_mixed_precision}")
            
            # Initialize DeepLabV3 model (use weights parameter instead of pretrained)
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
            
            # Adjust classifier for 2 classes (background + camouflage)
            self.model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
            
            # Load weights with safe globals for numpy compatibility
            print(f"   Loading checkpoint...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as load_error:
                # Fallback: Add safe globals for numpy if needed
                print(f"   Retrying with numpy safe globals...")
                import numpy as np
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            # Optimize model
            self.model.eval()
            self.model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Setup mixed precision if enabled
            if self.use_mixed_precision and self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
                print("   ✅ Mixed Precision (FP16) enabled")
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print("✅ Model loaded successfully!")
            
            # Get model size
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"   Model Parameters: {model_size:.2f}M")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_gpu_memory_usage(self):
        """Get GPU memory usage info"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return allocated, reserved
        return 0, 0
    
    def segmentAsPascalvoc(self, frame):
        """
        Segment frame and return results in PASCAL VOC format
        GPU optimized with mixed precision inference
        
        Args:
            frame: Input frame (BGR from OpenCV)
            
        Returns:
            results: Segmentation results dictionary
            output: Segmented frame with overlay
        """
        
        if self.model is None:
            print("❌ Model not loaded")
            return None, frame
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Resize for model
            frame_resized = cv2.resize(frame_rgb, self.input_size)
            pil_image = Image.fromarray(frame_resized)
            
            # Prepare input tensor
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # GPU inference with mixed precision
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = self.model(input_tensor)['out']
                else:
                    output = self.model(input_tensor)['out']
            
            # Get segmentation map
            segmentation_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            
            # Resize back to original size
            segmentation_map = cv2.resize(
                segmentation_map.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Create binary mask (class 1 = camouflage)
            binary_mask = (segmentation_map == 1).astype(np.uint8) * 255
            
            # Create colored segmentation output
            segmented_output = frame.copy()
            segmented_output[binary_mask > 0] = [0, 0, 255]  # Red for detections
            
            # Blend with original
            output_frame = cv2.addWeighted(frame, 0.6, segmented_output, 0.4, 0)
            
            # Find and draw contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detection_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    detection_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        output_frame,
                        "Camouflage",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            
            # Results dictionary
            results = {
                'detection_count': detection_count,
                'segmentation_map': segmentation_map,
                'binary_mask': binary_mask
            }
            
            return results, output_frame
            
        except Exception as e:
            print(f"❌ Error during segmentation: {e}")
            return None, frame


# ============================================================================
# Location Detection
# ============================================================================

def get_device_location() -> Dict[str, float]:
    """
    Get device location using Windows Location API (GPS/Geolocation like browser)
    Uses PowerShell to access Windows Location Services
    Falls back to IP geolocation if Windows Location is unavailable
    
    Returns:
        Dict with 'latitude' and 'longitude', or (0.0, 0.0) if unavailable
    """
    # Try Windows Location API first (if on Windows)
    if platform.system() == 'Windows':
        try:
            print("   📡 Requesting location from Windows Location API (GPS)...")
            
            # Use PowerShell to get Windows Location (like browser geolocation)
            ps_script = """
            Add-Type -AssemblyName System.Device
            $GeoWatcher = New-Object System.Device.Location.GeoCoordinateWatcher
            $GeoWatcher.Start()
            $maxWait = 10
            $counter = 0
            while (($GeoWatcher.Status -ne 'Ready') -and ($counter -lt $maxWait)) {
                Start-Sleep -Milliseconds 500
                $counter++
            }
            if ($GeoWatcher.Status -eq 'Ready') {
                $location = $GeoWatcher.Position.Location
                Write-Output "$($location.Latitude),$($location.Longitude)"
            } else {
                Write-Output "FAILED"
            }
            $GeoWatcher.Stop()
            """
            
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and result.stdout.strip() != "FAILED":
                coords = result.stdout.strip().split(',')
                if len(coords) == 2:
                    try:
                        latitude = float(coords[0])
                        longitude = float(coords[1])
                        
                        # Verify valid coordinates
                        if abs(latitude) <= 90 and abs(longitude) <= 180 and latitude != 0.0 and longitude != 0.0:
                            print(f"   ✅ GPS Location detected: ({latitude:.6f}, {longitude:.6f})")
                            print(f"   📡 Source: Windows Location Services (GPS/WiFi)")
                            return {
                                'latitude': latitude,
                                'longitude': longitude
                            }
                    except ValueError:
                        pass
            
            # Windows Location not available
            print("   ⚠️  Windows Location Services unavailable or permission denied")
            print("   💡 Enable Location Services in Windows Settings > Privacy > Location")
            print("   📡 Falling back to IP-based geolocation...")
            
        except subprocess.TimeoutExpired:
            print("   ⚠️  Windows Location request timed out")
            print("   📡 Falling back to IP-based geolocation...")
        except FileNotFoundError:
            print("   ⚠️  PowerShell not found, using IP geolocation...")
        except Exception as e:
            print(f"   ⚠️  Windows Location API error: {e}")
            print("   📡 Falling back to IP-based geolocation...")
    
    # Fallback: IP-based geolocation (works on all platforms)
    try:
        print("   📡 Requesting location from IP geolocation service...")
        response = requests.get('http://ip-api.com/json/', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                latitude = float(data.get('lat', 0.0))
                longitude = float(data.get('lon', 0.0))
                
                # Verify valid coordinates
                if latitude != 0.0 and longitude != 0.0:
                    city = data.get('city', 'Unknown')
                    country = data.get('country', 'Unknown')
                    print(f"   ✅ Location detected: ({latitude:.6f}, {longitude:.6f})")
                    print(f"   📍 City: {city}, Country: {country} (IP-based)")
                    return {
                        'latitude': latitude,
                        'longitude': longitude
                    }
                else:
                    print(f"   ⚠️  Invalid coordinates received: lat={latitude}, lon={longitude}")
    except requests.exceptions.Timeout:
        print("   ⚠️  Location request timed out")
    except requests.exceptions.ConnectionError:
        print("   ⚠️  Could not connect to location service (check internet connection)")
    except Exception as e:
        print(f"   ⚠️  Error detecting location: {e}")
    
    print("   ⚠️  Using default location (0.0, 0.0)")
    return {
        'latitude': 0.0,
        'longitude': 0.0
    }


# ============================================================================
# Report Generation and Database Integration
# ============================================================================

class ReportGenerator:
    """Handles report generation and Firebase integration"""
    
    def __init__(self):
        self.firestore_handler = FirestoreHandler()
        self.firebase_storage = FirebaseStorageHandler()
        self.llm_generator = None  # Will be initialized after Firebase
        self.last_report_time = None  # Track when last report was sent
        self.report_interval = 120  # 2 minutes in seconds
        
    def initialize(self):
        """Initialize Firebase handlers and LLM"""
        try:
            print("\n📡 Initializing Firebase connection...")
            
            # Initialize Firestore
            if not self.firestore_handler.initialize():
                print("⚠️  Firestore initialization failed, but continuing...")
                return False
            
            # Initialize Firebase Storage
            if not self.firebase_storage.initialize():
                print("⚠️  Firebase Storage initialization failed, but continuing...")
                return False
            
            print("✅ Firebase initialized successfully!")
            
            # Initialize LLM after Firebase (so environment is fully loaded)
            print("\n🤖 Initializing OpenAI LLM handler...")
            self.llm_generator = LLMReportGenerator()
            if self.llm_generator.check_connection():
                print("✅ OpenAI LLM ready for AI analysis")
            else:
                print("⚠️  OpenAI API key not configured - reports will have limited analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
            return False
    
    def frame_to_base64(self, frame) -> str:
        """Convert OpenCV frame to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            print(f"❌ Error converting frame to base64: {e}")
            return ""
    
    def frame_to_pil(self, frame) -> Image.Image:
        """Convert OpenCV BGR frame to PIL Image"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            return pil_image
        except Exception as e:
            print(f"❌ Error converting frame to PIL: {e}")
            return None
    
    def can_generate_report(self) -> bool:
        """
        Check if enough time has passed since last report
        
        Returns:
            bool: True if 2 minutes have passed since last report, or if no report has been sent yet
        """
        if self.last_report_time is None:
            return True
        
        time_elapsed = time.time() - self.last_report_time
        return time_elapsed >= self.report_interval
    
    def get_time_until_next_report(self) -> int:
        """
        Get seconds remaining until next report can be generated
        
        Returns:
            int: Seconds remaining (0 if ready to report)
        """
        if self.last_report_time is None:
            return 0
        
        time_elapsed = time.time() - self.last_report_time
        remaining = max(0, self.report_interval - time_elapsed)
        return int(remaining)
    
    async def generate_and_send_report(self, frame, segmented_frame, detection_count: int, auto_mode: bool = False):
        """
        Generate report from frame and send to database
        Respects 2-minute interval between reports to prevent spamming
        
        Args:
            frame: Original OpenCV frame with detection
            segmented_frame: Segmented overlay frame
            detection_count: Number of camouflaged soldiers detected
            auto_mode: If True, this is an automatic report generation
        
        Returns:
            bool: True if report was generated, False otherwise
        """
        # Check if enough time has passed since last report
        if not self.can_generate_report():
            time_remaining = self.get_time_until_next_report()
            if auto_mode:
                # Silent skip in auto mode
                pass
            else:
                print(f"⚠️  Report cooldown active. Next report available in {time_remaining} seconds ({time_remaining//60}m {time_remaining%60}s)")
            return False
        
        if detection_count == 0:
            print("⚠️  No detections found, skipping report generation")
            return False
        
        try:
            print("\n" + "=" * 70)
            print("🔍 GENERATING DETECTION REPORT")
            print("=" * 70)
            
            # Step 1: Get AI Analysis from OpenAI Vision
            print("\n1️⃣  Getting AI analysis from OpenAI GPT-4 Vision API...")
            pil_frame = self.frame_to_pil(frame)
            if pil_frame is None:
                print("❌ Failed to convert frame")
                return False
            
            # Check if LLM is available
            if self.llm_generator is None:
                print("❌ LLM generator not initialized - cannot generate AI analysis")
                return False
            
            ai_analysis = await self.llm_generator.generate_report(pil_frame)
            print(f"   Environment: {ai_analysis.get('environment', 'Unknown')}")
            print(f"   Soldiers detected: {ai_analysis.get('camouflaged_soldier_count', 0)}")
            print(f"   Attire: {ai_analysis.get('attire_and_camouflage', 'Unknown')}")
            print(f"   Equipment: {ai_analysis.get('equipment', 'Unknown')}")
            
            # Step 2: Generate report ID ONCE (used for both image upload and Firestore document)
            print("\n2️⃣  Generating unique report ID...")
            if self.firestore_handler._initialized:
                report_id = self.firestore_handler.generate_report_id()
            else:
                # Fallback to timestamp-based ID
                report_id = f"LIVE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            print(f"   Report ID: {report_id}")
            
            # Step 3: Upload images to Firebase Storage (gs://mirqab-9de3f.firebasestorage.app)
            print("\n3️⃣  Uploading images to Firebase Storage...")
            original_url = None
            segmented_url = None
            
            if self.firebase_storage._initialized:
                # Upload original frame
                original_base64 = self.frame_to_base64(frame)
                if original_base64:
                    original_url = self.firebase_storage.upload_image(
                        original_base64,
                        report_id,
                        "original"
                    )
                    if original_url:
                        print(f"   ✅ Original image uploaded to: {original_url[:80]}...")
                
                # Upload segmented overlay frame
                segmented_base64 = self.frame_to_base64(segmented_frame)
                if segmented_base64:
                    segmented_url = self.firebase_storage.upload_image(
                        segmented_base64,
                        report_id,
                        "segmented"
                    )
                    if segmented_url:
                        print(f"   ✅ Segmented image uploaded to: {segmented_url[:80]}...")
            else:
                print("   ⚠️  Firebase Storage not initialized - saving locally only")
                # Save frames locally as backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                local_original = f"camouflage_detection_{timestamp}_original.jpg"
                local_segmented = f"camouflage_detection_{timestamp}_segmented.jpg"
                cv2.imwrite(local_original, frame)
                cv2.imwrite(local_segmented, segmented_frame)
                print(f"   💾 Saved locally: {local_original}, {local_segmented}")
            
            # Step 4: Get device location (like upload page does)
            print("\n4️⃣  Detecting device location...")
            device_location = get_device_location()
            
            # Debug: Print detected location
            print(f"   📊 Detected coordinates: lat={device_location['latitude']:.6f}, lon={device_location['longitude']:.6f}")
            
            # Step 5: Create detection report in Firestore
            print("\n5️⃣  Saving report to Firestore database...")
            
            # Prepare report data with detected location
            report_data = {
                "report_id": report_id,
                "location": {
                    "longitude": float(device_location["longitude"]),  # Ensure it's a float
                    "latitude": float(device_location["latitude"])       # Ensure it's a float
                },
                "environment": ai_analysis.get("environment", "Unknown"),
                "soldier_count": ai_analysis.get("camouflaged_soldier_count", detection_count),
                "attire_and_camouflage": ai_analysis.get("attire_and_camouflage", "Unknown"),
                "equipment": ai_analysis.get("equipment", "Unknown"),
                "source_device_id": "Live-Camera-Detector",
                "image_snapshot_url": original_url or "",
                "segmented_image_url": segmented_url or ""
            }
            
            if self.firestore_handler._initialized:
                # Use the SAME report_id for Firestore document (don't generate a new one)
                from firebase_admin import firestore as fb_firestore
                
                # Add timestamp for Firestore
                report_data['timestamp'] = fb_firestore.SERVER_TIMESTAMP
                
                doc_ref = self.firestore_handler.db.collection('detection_reports').document(report_id)
                doc_ref.set(report_data)
                
                print(f"   ✅ Report saved to Firestore: {report_id}")
                print(f"   📍 Location: ({report_data['location']['latitude']:.6f}, {report_data['location']['longitude']:.6f})")
                if device_location['latitude'] != 0.0 or device_location['longitude'] != 0.0:
                    print(f"   🌍 Real device location detected and saved")
                else:
                    print(f"   ⚠️  Default location used (0.0, 0.0) - location detection unavailable")
                print(f"   🖼️  Original image: {bool(original_url)}")
                print(f"   🎨 Segmented image: {bool(segmented_url)}")
                print("\n" + "=" * 70)
                print("✅ DETECTION REPORT SUCCESSFULLY GENERATED AND STORED")
                print("=" * 70)
                
                # Update last report time
                self.last_report_time = time.time()
                
                if auto_mode:
                    print(f"   ⏰ Auto-report mode: Next report available in {self.report_interval} seconds (2 minutes)")
                
                return True
            else:
                print("   ⚠️  Firestore not initialized, report not saved to database")
                print("   📝 Report data prepared (local only):")
                import json
                print(f"      {json.dumps(report_data, indent=2)}")
                return False
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            traceback.print_exc()
            return False


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("Real-time Segmentation with Custom PyTorch Model")
    print("🚀 GPU OPTIMIZED VERSION")
    print("=" * 70)

    # Initialize segmentation with GPU optimization
    use_mixed_precision = True  # FP16 for faster inference
    segment_image = CustomSemanticSegmentation(use_mixed_precision=use_mixed_precision)

    # Load your custom model
    model_loaded = segment_image.load_model("best_deeplabv3_camouflage.pth")

    if not model_loaded:
        print("\n❌ Failed to load model. Exiting...")
        exit(1)

    # Initialize report generator
    print("\n" + "=" * 70)
    print("Initializing Report Generator...")
    print("=" * 70)
    report_gen = ReportGenerator()
    firebase_available = report_gen.initialize()

    print("\n" + "=" * 70)
    print("Starting real-time segmentation with AUTO-REPORT...")
    print("📡 Reports will be sent automatically every 2 minutes when camouflage is detected")
    print("Press 'q' to quit, 's' to save frame, 'g' to manually generate report")
    print("=" * 70 + "\n")

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Failed to open webcam")
        exit(1)

    frame_count = 0
    detection_frame = None
    segmented_frame = None
    fps_counter = 0
    last_time = time.time()
    
    # For auto-report tracking
    last_auto_report_attempt = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Segment frame (GPU accelerated)
            results, output = segment_image.segmentAsPascalvoc(frame)
            
            if output is not None:
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time
                
                # Add statistics
                if results:
                    detection_count = results['detection_count']
                    
                    # GPU memory usage
                    gpu_allocated, gpu_reserved = segment_image.get_gpu_memory_usage()
                    
                    # Display info
                    info_text = f"Detections: {detection_count} | Frame: {frame_count} | FPS: {fps:.1f}"
                    if segment_image.device.type == 'cuda':
                        info_text += f" | GPU: {gpu_allocated:.1f}GB"
                    
                    cv2.putText(
                        output,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Show device info
                    device_text = f"Device: {segment_image.device.type.upper()}"
                    if segment_image.device.type == 'cuda':
                        device_text += f" | Mixed Precision: ON"
                    
                    cv2.putText(
                        output,
                        device_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (100, 200, 255),
                        2
                    )
                    
                    # Auto-report countdown display
                    time_remaining = report_gen.get_time_until_next_report()
                    if time_remaining > 0:
                        countdown_text = f"Next auto-report: {time_remaining//60}m {time_remaining%60}s"
                        text_color = (0, 165, 255)  # Orange
                    else:
                        countdown_text = "Auto-report: READY"
                        text_color = (0, 255, 0)  # Green
                    
                    cv2.putText(
                        output,
                        countdown_text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        text_color,
                        2
                    )
                    
                    # Auto-generate report when camouflage detected (every 2 minutes)
                    if detection_count > 0:
                        # Store both frames for reporting
                        detection_frame = frame.copy()
                        segmented_frame = output.copy()
                        
                        # Try to auto-generate report if cooldown is over
                        if report_gen.can_generate_report():
                            print(f"\n🔔 Auto-generating report - Camouflage detected ({detection_count} targets)")
                            asyncio.run(report_gen.generate_and_send_report(detection_frame, segmented_frame, detection_count, auto_mode=True))
                
                # Display
                cv2.imshow("Real-time Segmentation - Press 'q' to quit, 'g' to generate report", output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✅ Exiting...")
                    break
                elif key == ord('s'):
                    filename = f"segmentation_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, output)
                    print(f"💾 Saved: {filename}")
                elif key == ord('g'):
                    # Manual report generation (bypasses auto timer if needed)
                    if detection_frame is not None and segmented_frame is not None:
                        print(f"\n🔔 Manual report generation triggered...")
                        asyncio.run(report_gen.generate_and_send_report(detection_frame, segmented_frame, results['detection_count'], auto_mode=False))
                    else:
                        print("⚠️  No detection frame available. Camouflage must be detected first.")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    finally:
        cv2.destroyAllWindows()
        camera.release()
        
        # Clean up GPU memory
        if segment_image.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 70)
        print(f"Total frames processed: {frame_count}")
        print(f"Auto-report enabled: Yes (every 2 minutes)")
        print(f"Device used: {segment_image.device.type.upper()}")
        print("Session ended")
        print("=" * 70)


if __name__ == "__main__":
    main()