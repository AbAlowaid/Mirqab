#!/usr/bin/env python3
"""
Mirqab Raspberry Pi Camera Client
Captures frames from Pi Camera and sends to laptop server for AI processing
✅ Real-time camouflage detection with remote AI processing
"""

import cv2
import numpy as np
import requests
import base64
import time
import json
from datetime import datetime
import argparse

class RaspberryPiClient:
    def __init__(self, server_url="http://192.168.1.100:8081"):
        """
        Initialize Raspberry Pi camera client
        
        Args:
            server_url: URL of the laptop processing server
        """
        self.server_url = server_url
        self.health_endpoint = f"{server_url}/health"
        self.process_endpoint = f"{server_url}/process_frame"
        self.camera = None
        self.frame_count = 0
        self.detection_count = 0
        self.report_count = 0
        
    def check_server_health(self):
        """Check if the server is healthy and ready"""
        try:
            print(f"🔍 Checking server health at {self.health_endpoint}...")
            response = requests.get(self.health_endpoint, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is healthy!")
                print(f"   Model loaded: {data.get('model_loaded', False)}")
                print(f"   Device: {data.get('device', 'Unknown')}")
                return True
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to server at {self.server_url}")
            print(f"   Make sure the server is running on the laptop")
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def initialize_camera(self, camera_index=0, width=640, height=480):
        """
        Initialize camera (works with USB camera or Pi Camera via V4L2)
        
        Args:
            camera_index: Camera device index (0 for default)
            width: Frame width
            height: Frame height
        """
        try:
            print(f"\n📷 Initializing camera {camera_index}...")
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                print(f"❌ Failed to open camera {camera_index}")
                return False
            
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                print(f"❌ Failed to capture test frame")
                return False
            
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Camera initialized successfully!")
            print(f"   Resolution: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
    
    def encode_frame(self, frame):
        """Encode frame to base64 for transmission"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        except Exception as e:
            print(f"❌ Error encoding frame: {e}")
            return None
    
    def decode_frame(self, frame_base64):
        """Decode base64 frame to numpy array"""
        try:
            frame_data = base64.b64decode(frame_base64)
            frame_np = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"❌ Error decoding frame: {e}")
            return None
    
    def send_frame_for_processing(self, frame):
        """
        Send frame to laptop server for AI processing
        
        Returns:
            dict: Processing results including mask, detection info, and report status
        """
        try:
            # Encode frame
            frame_base64 = self.encode_frame(frame)
            if frame_base64 is None:
                return None
            
            # Prepare request
            payload = {
                "frame": frame_base64,
                "timestamp": time.time()
            }
            
            # Send to server
            response = requests.post(
                self.process_endpoint,
                json=payload,
                timeout=30  # Increased timeout for AI processing
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Server returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⚠️  Request timeout - server may be busy")
            return None
        except requests.exceptions.ConnectionError:
            print(f"⚠️  Connection lost to server")
            return None
        except Exception as e:
            print(f"❌ Error sending frame: {e}")
            return None
    
    def run_detection(self, display=True, save_detections=False, fps_limit=2):
        """
        Run real-time detection loop
        
        Args:
            display: Show video feed (requires X11/display)
            save_detections: Save frames when detection occurs
            fps_limit: Max frames per second to process (reduces bandwidth)
        """
        if self.camera is None:
            print("❌ Camera not initialized")
            return
        
        print("\n" + "=" * 70)
        print("🚀 Starting Real-time Detection")
        print("=" * 70)
        print(f"📡 Server: {self.server_url}")
        print(f"📹 FPS Limit: {fps_limit} (processing rate)")
        print(f"💾 Save detections: {'Yes' if save_detections else 'No'}")
        print(f"🖥️  Display: {'Yes' if display else 'No (headless mode)'}")
        print("⏹️  Press 'q' to quit")
        print("=" * 70 + "\n")
        
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0
        last_process_time = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Rate limiting
                current_time = time.time()
                if current_time - last_process_time < frame_delay:
                    if display:
                        cv2.imshow("Raspberry Pi - Mirqab Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                
                last_process_time = current_time
                
                # Send frame for processing
                print(f"\r📸 Frame {self.frame_count} - Processing...", end="", flush=True)
                result = self.send_frame_for_processing(frame)
                
                if result:
                    detection_info = result.get('detection', {})
                    detected = detection_info.get('detected', False)
                    count = detection_info.get('count', 0)
                    confidence = detection_info.get('confidence', 0.0)
                    report_generated = result.get('report_generated', False)
                    cooldown = result.get('cooldown_remaining', 0)
                    
                    # Update stats
                    if detected:
                        self.detection_count += 1
                        if report_generated:
                            self.report_count += 1
                    
                    # Display results
                    status = "🔴 DETECTED" if detected else "✅ Clear"
                    print(f"\r📸 Frame {self.frame_count} - {status} | Count: {count} | Conf: {confidence:.2f} | Reports: {self.report_count} | Cooldown: {cooldown}s", flush=True)
                    
                    # Decode and display overlay
                    if display and 'overlay' in result:
                        overlay_frame = self.decode_frame(result['overlay'])
                        if overlay_frame is not None:
                            # Add status text
                            info_text = f"Detections: {count} | Reports: {self.report_count}"
                            cv2.putText(overlay_frame, info_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if cooldown > 0:
                                cooldown_text = f"Next report: {cooldown}s"
                                cv2.putText(overlay_frame, cooldown_text, (10, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            
                            cv2.imshow("Raspberry Pi - Mirqab Detection", overlay_frame)
                    
                    # Save detection if requested
                    if save_detections and detected and 'overlay' in result:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"detection_{timestamp}_{count}targets.jpg"
                        overlay_frame = self.decode_frame(result['overlay'])
                        if overlay_frame is not None:
                            cv2.imwrite(filename, overlay_frame)
                            print(f"\n💾 Saved: {filename}")
                
                else:
                    print(f"\r📸 Frame {self.frame_count} - ⚠️  No response from server", flush=True)
                
                # Check for quit key
                if display:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n✅ Quit requested by user")
                        break
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Total frames captured: {self.frame_count}")
        print(f"Detections: {self.detection_count}")
        print(f"Reports generated: {self.report_count}")
        print("=" * 70)


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Camera Client for Mirqab Detection System"
    )
    parser.add_argument(
        "--server", "-s",
        default="http://192.168.1.100:8081",
        help="Laptop server URL (default: http://192.168.1.100:8081)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height", "-h",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Max frames per second to process (default: 2)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode (no video display)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save frames when detection occurs"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🍓 Mirqab Raspberry Pi Camera Client")
    print("=" * 70)
    
    # Initialize client
    client = RaspberryPiClient(server_url=args.server)
    
    # Check server health
    if not client.check_server_health():
        print("\n❌ Server health check failed. Please start the server first.")
        print(f"   Run this on your laptop: python raspberry.py")
        return
    
    # Initialize camera
    if not client.initialize_camera(args.camera, args.width, args.height):
        print("\n❌ Camera initialization failed.")
        return
    
    # Run detection
    client.run_detection(
        display=not args.no_display,
        save_detections=args.save,
        fps_limit=args.fps
    )


if __name__ == "__main__":
    main()


