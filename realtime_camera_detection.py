"""
Real-time Camera Detection with Model Masking Overlay using ImageIO
Uses custom PyTorch DeepLabV3 model for camouflage detection
"""

import time
import sys
import os
import argparse
from typing import Optional, Tuple, Dict, Any

import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import imageio.v3 as iio


class RealTimeDetector:
    """
    Real-time camouflage detection using imageio camera and PyTorch model
    """
    
    def __init__(self, model_path: str = "best_deeplabv3_camouflage.pth", 
                 input_size: Tuple[int, int] = (640, 480),
                 confidence_threshold: float = 0.5,
                 min_detection_area: int = 500):
        """
        Initialize the real-time detector
        
        Args:
            model_path: Path to the PyTorch model file
            input_size: Input size for the model (width, height)
            confidence_threshold: Minimum confidence for detections
            min_detection_area: Minimum area for valid detections
        """
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.min_detection_area = min_detection_area
        
        self.model = None
        self.transform = None
        self.device = None
        self.camera = None
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        self.fps_history = []
        
        # Detection colors
        self.colors = {
            'detection': (0, 255, 0),      # Green for bounding boxes
            'mask': (0, 0, 255),          # Red for mask overlay
            'text': (255, 255, 255),      # White for text
            'background': (0, 0, 0)       # Black for background
        }
    
    def load_model(self) -> bool:
        """
        Load the PyTorch model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"📦 Loading model from: {self.model_path}")
            print(f"   Device: {self.device.upper()}")
            
            # Initialize DeepLabV3 model
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
            
            # Adjust classifier for 2 classes (background + camouflage)
            self.model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            self.model.to(self.device)
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def open_camera(self, camera_index: int = 0) -> bool:
        """
        Open camera using imageio
        
        Args:
            camera_index: Camera index to use
            
        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        try:
            print(f"🎥 Opening camera {camera_index}...")
            self.camera = iio.imopen(f"<video{camera_index}>", "r")
            
            # Test read a frame
            test_frame = self.camera.read()
            print(f"✅ Camera opened successfully!")
            print(f"   Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            print(f"   Channels: {test_frame.shape[2]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error opening camera: {e}")
            return False
    
    def detect_and_mask(self, frame: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        Perform detection and create mask overlay
        
        Args:
            frame: Input frame (RGB from imageio)
            
        Returns:
            Tuple of (detection_results, output_frame_with_overlay)
        """
        if self.model is None:
            return None, frame
        
        try:
            h, w = frame.shape[:2]
            
            # Resize for model
            frame_resized = cv2.resize(frame, self.input_size)
            pil_image = Image.fromarray(frame_resized)
            
            # Inference
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
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
            
            # Create mask overlay
            mask_overlay = frame.copy()
            mask_overlay[binary_mask > 0] = self.colors['mask']
            
            # Blend mask with original frame
            output_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
            
            # Find contours and draw bounding boxes
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_detection_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Draw bounding box
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), 
                                self.colors['detection'], 2)
                    
                    # Draw label
                    label = f"Camouflage ({area:.0f}px)"
                    cv2.putText(output_frame, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              self.colors['detection'], 2)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': 1.0  # Binary segmentation
                    })
            
            # Results dictionary
            results = {
                'detection_count': len(detections),
                'detections': detections,
                'segmentation_map': segmentation_map,
                'binary_mask': binary_mask
            }
            
            return results, output_frame
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return None, frame
    
    def add_info_overlay(self, frame: np.ndarray, results: Optional[Dict]) -> np.ndarray:
        """
        Add information overlay to the frame
        
        Args:
            frame: Input frame
            results: Detection results
            
        Returns:
            Frame with information overlay
        """
        overlay_frame = frame.copy()
        
        # Calculate FPS
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:  # Keep last 30 FPS readings
                self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
        else:
            avg_fps = 0
        
        # Add statistics
        info_lines = [
            f"Frame: {self.frame_count}",
            f"FPS: {avg_fps:.1f}",
            f"Detections: {results['detection_count'] if results else 0}",
            f"Device: {self.device.upper()}"
        ]
        
        # Draw background rectangle
        cv2.rectangle(overlay_frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(overlay_frame, line, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['text'], 2)
        
        # Add controls info
        controls_text = "Press 'q' to quit, 's' to save, 'r' to reset stats"
        cv2.putText(overlay_frame, controls_text, (10, overlay_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return overlay_frame
    
    def save_frame(self, frame: np.ndarray, prefix: str = "detection") -> str:
        """
        Save current frame to file
        
        Args:
            frame: Frame to save
            prefix: Filename prefix
            
        Returns:
            str: Saved filename
        """
        timestamp = int(time.time())
        filename = f"{prefix}_frame_{self.frame_count}_{timestamp}.jpg"
        
        # Convert RGB to BGR for saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame_bgr)
        
        return filename
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        self.fps_history = []
        print("📊 Statistics reset")
    
    def run_detection(self, camera_index: int = 0, save_interval: int = 0):
        """
        Run real-time detection
        
        Args:
            camera_index: Camera index to use
            save_interval: Save frame every N frames (0 = manual only)
        """
        print("=" * 70)
        print("Real-time Camera Detection with Model Masking Overlay")
        print("=" * 70)
        
        # Load model
        if not self.load_model():
            print("❌ Failed to load model. Exiting...")
            return
        
        # Open camera
        if not self.open_camera(camera_index):
            print("❌ Failed to open camera. Exiting...")
            return
        
        print("\n" + "=" * 70)
        print("Starting real-time detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset statistics")
        print("=" * 70 + "\n")
        
        self.start_time = time.time()
        
        try:
            while True:
                # Read frame
                frame = self.camera.read()
                if frame is None:
                    print("❌ Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Perform detection
                results, output_frame = self.detect_and_mask(frame)
                
                # Add information overlay
                final_frame = self.add_info_overlay(output_frame, results)
                
                # Convert RGB to BGR for OpenCV display
                display_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow("Real-time Detection - Press 'q' to quit", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✅ Exiting...")
                    break
                elif key == ord('s'):
                    filename = self.save_frame(final_frame)
                    print(f"💾 Saved: {filename}")
                elif key == ord('r'):
                    self.reset_stats()
                
                # Auto-save if interval is set
                if save_interval > 0 and self.frame_count % save_interval == 0:
                    filename = self.save_frame(final_frame, "auto")
                    print(f"💾 Auto-saved: {filename}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.close()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Total detections: {self.detection_count}")
            print("=" * 70)


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Real-time camera detection with model masking overlay"
    )
    parser.add_argument(
        "--model", "-m",
        default="best_deeplabv3_camouflage.pth",
        help="Path to PyTorch model file"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index to use"
    )
    parser.add_argument(
        "--size", "-s",
        nargs=2,
        type=int,
        default=[640, 480],
        help="Input size for model (width height)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum detection area in pixels"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save frame every N frames (0 = manual only)"
    )
    
    args = parser.parse_args()
    
    # Create detector
    detector = RealTimeDetector(
        model_path=args.model,
        input_size=tuple(args.size),
        min_detection_area=args.min_area
    )
    
    # Run detection
    detector.run_detection(
        camera_index=args.camera,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()


