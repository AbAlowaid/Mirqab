"""
Model Handler - Load and run the DeepLabV3 semantic segmentation model
PyTorch-based segmentation with camouflage detection
"""

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

class SegmentationModel:
    def __init__(self, model_path: str = "../best_deeplabv3_camouflage.pth", config_path: str = "../data.yaml"):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.class_names = {0: "background", 1: "camouflage_soldier"}
        self.num_classes = 2  # background + camouflage_soldier
        self._loaded = False
        
        # Define preprocessing transforms for DeepLabV3
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Using device: {self.device}")
    
    def load_model(self):
        try:
            print(f"Loading DeepLabV3 model...")
            print(f"   Model path: {self.model_path}")
            
            # Create DeepLabV3 model with ResNet-101 backbone (model was trained with ResNet-101)
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
            
            # Modify classifier to match number of classes
            self.model.classifier[4] = torch.nn.Conv2d(256, self.num_classes, kernel_size=1)
            
            # Modify aux_classifier if it exists
            if self.model.aux_classifier is not None:
                self.model.aux_classifier[4] = torch.nn.Conv2d(256, self.num_classes, kernel_size=1)
            
            # Load trained weights
            if self.model_path.exists():
                # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
                try:
                    checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
                except Exception as load_error:
                    # Fallback: Add safe globals for numpy if needed
                    print(f"   Retrying with numpy safe globals...")
                    import numpy as np
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                    checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Load with strict=False to allow missing or extra keys (e.g., aux_classifier)
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                    
                print("   Note: Loaded with strict=False (auxiliary classifier may differ)")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            print("DeepLabV3 model loaded successfully!")
            print(f"   - Device: {self.device}")
            print(f"   - Backbone: ResNet-101")
            print(f"   - Number of classes: {self.num_classes}")
            print(f"   - Class names: {self.class_names}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def is_loaded(self):
        return self._loaded
    
    def predict(self, image):
        """
        Predict segmentation mask for a single image.
        Returns binary mask and segmentation map.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Get original dimensions
            original_width, original_height = image.size
            
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)['out']
            
            # Get segmentation map (class predictions for each pixel)
            segmentation_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            
            # Resize to original image size
            segmentation_map = cv2.resize(
                segmentation_map.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Create binary mask for soldier class (class_id = 1)
            soldier_class_id = 1
            binary_mask = (segmentation_map == soldier_class_id).astype(np.uint8)
            
            return binary_mask, segmentation_map
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty masks on error
            height, width = image.size[1], image.size[0]
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            empty_seg_map = np.zeros((height, width), dtype=np.int32)
            return empty_mask, empty_seg_map
    
    def predict_with_details(self, image):
        """
        Predict with detailed instance extraction using connected components.
        Returns mask, instances, and counts.
        """
        binary_mask, segmentation_map = self.predict(image)
        
        detections = []
        soldier_count = 0
        
        # Extract individual soldier instances using connected components
        if np.any(binary_mask > 0):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            
            # Minimum area threshold to filter noise
            min_area = 100
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area >= min_area:
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    y = int(stats[i, cv2.CC_STAT_TOP])
                    w = int(stats[i, cv2.CC_STAT_WIDTH])
                    h = int(stats[i, cv2.CC_STAT_HEIGHT])
                    
                    # Create instance mask
                    instance_mask = (labels == i).astype(np.uint8)
                    
                    detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "score": 0.95,
                        "confidence": 0.95,
                        "mask": instance_mask,
                        "class_id": 1,
                        "class_name": "camouflage_soldier"
                    })
                    soldier_count += 1
        
        return {
            "mask": binary_mask,
            "instances": detections,
            "soldier_count": soldier_count,
            "civilian_count": 0,
            "total_count": soldier_count,
            "count": soldier_count
        }
    
    def predict_batch(self, images):
        """
        Predict on a batch of images.
        Returns list of (mask, segmentation_map) tuples.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for image in images:
            mask, seg_map = self.predict(image)
            results.append((mask, seg_map))
        
        return results
