
import os
import sys
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Environment variables (can be set via os.environ)
YOLO_DETECTION_ENABLED = os.getenv("YOLO_DETECTION_ENABLED", "true").lower() == "true"
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "s").lower()  # n, s, m, l, x
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

# ============================================================================
# Global Model Variables
# ============================================================================

_yolo_model = None
_yolo_model_loaded = False

# ============================================================================
# Model Loading
# ============================================================================

def load_yolo_model():
    """Load YOLOv8 model for object detection."""
    global _yolo_model, _yolo_model_loaded
    
    if _yolo_model_loaded:
        return True
    
    print(" Loading YOLOv8 model...")
    
    if not YOLO_DETECTION_ENABLED:
        print("  YOLO detection is disabled")
        return False
    
    try:
        from ultralytics import YOLO
        model_name = f"yolov8{YOLO_MODEL_SIZE}.pt"
        _yolo_model = YOLO(model_name)
        _yolo_model_loaded = True
        print(f" YOLOv8 model loaded ({model_name})")
        return True
    except Exception as e:
        print(f"  Warning: YOLOv8 not available: {e}")
        _yolo_model = None
        return False

# ============================================================================
# Image Preprocessing
# ============================================================================

def preprocess_image(img_bytes: bytes) -> Image.Image:
    
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize for performance (800px max dimension)
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    
    # Enhance image quality
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    
    return img

def compute_image_quality_score(img: Image.Image) -> Tuple[float, Dict]:
    """
    Compute image quality score (0.0-1.0) based on blur, brightness, contrast.
    
    Returns:
        quality_score: float in [0.0, 1.0]
        quality_metrics: dict with individual scores
    """
    try:
        import cv2
        
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        metrics = {}
        
        # Blur score (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 200.0)
        metrics['blur_score'] = blur_score
        
        # Brightness score
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            brightness_score = max(0.0, mean_brightness / 50.0)
        elif mean_brightness > 230:
            brightness_score = max(0.0, (255 - mean_brightness) / 25.0)
        else:
            brightness_score = 1.0
        metrics['brightness_score'] = brightness_score
        
        # Contrast score
        contrast_std = np.std(gray)
        contrast_score = min(1.0, max(0.0, contrast_std / 50.0))
        metrics['contrast_score'] = contrast_score
        
        # Combined quality score
        quality_score = (blur_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        
        return quality_score, metrics
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Image quality scoring failed: {e}")
        return 0.8, {}

# ============================================================================
# YOLO Object Detection
# ============================================================================

def detect_objects_with_yolo(img: Image.Image) -> List[Dict]:
    """
    Detect objects using YOLOv8.
    
    Returns:
        List of detected objects with bbox, confidence, class_id, class_name
    """
    global _yolo_model
    
    detections = []
    
    if not _yolo_model or not YOLO_DETECTION_ENABLED:
        return detections
    
    try:
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]
        image_area = img_height * img_width
        min_area = 0.01 * image_area  # Minimum 1% of image area
        
        # YOLO detection with optimized thresholds for pantry scenes
        yolo_conf_threshold = 0.15  # Standard threshold
        yolo_iou_threshold = 0.50
        
        import torch
        with torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad():
            results = _yolo_model(img_array, conf=yolo_conf_threshold, 
                                iou=yolo_iou_threshold, verbose=False, max_det=50)
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        
                        # Filter small boxes
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area < min_area:
                            continue
                        
                        # Filter thin boxes
                        min_dimension = min(x2 - x1, y2 - y1)
                        if min_dimension < 0.005 * min(img_width, img_height):
                            continue
                        
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = _yolo_model.names[class_id] if hasattr(_yolo_model, 'names') else f"class_{class_id}"
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': class_id,
                            'class_name': class_name
                        })
                    except Exception as e:
                        if VERBOSE_LOGGING:
                            print(f"Warning: Error processing YOLO box: {e}")
                        continue
        
        return detections
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: YOLO detection failed: {e}")
        return []

# ============================================================================
# Food Item Mapping
# ============================================================================

def map_yolo_class_to_food_item(class_name: str) -> Optional[str]:
    """
    Map YOLO class names to food item names.
    This is a simple mapping - YOLO doesn't know food names, so we map object types.
    """
    class_lower = class_name.lower()
    
    # COCO classes that might be food-related
    food_mapping = {
        # Direct food items
        'banana': 'banana',
        'apple': 'apple',
        'orange': 'orange',
        'broccoli': 'broccoli',
        'carrot': 'carrot',
        'hot dog': 'hot dog',
        'pizza': 'pizza',
        'donut': 'donut',
        'cake': 'cake',
        'sandwich': 'sandwich',
        
        # Containers that might hold food
        'bottle': 'beverage',
        'cup': 'beverage',
        'bowl': 'food item',
        'vase': 'container',  # Sometimes misidentified
        
        # Utensils (not food, but might indicate food nearby)
        'fork': None,  # Skip utensils
        'knife': None,
        'spoon': None,
    }
    
    # Check direct mapping
    if class_lower in food_mapping:
        return food_mapping[class_lower]
    
    # Default: return class name as-is (might be food-related)
    return class_name

def categorize_item(item_name: str) -> str:
    """Categorize item based on name."""
    name_lower = item_name.lower()
    
    if any(x in name_lower for x in ['apple', 'banana', 'orange', 'broccoli', 'carrot']):
        return 'produce'
    elif any(x in name_lower for x in ['beverage', 'bottle', 'cup', 'juice', 'soda']):
        return 'beverages'
    elif any(x in name_lower for x in ['pizza', 'sandwich', 'hot dog', 'donut', 'cake']):
        return 'bakery'
    elif any(x in name_lower for x in ['bowl', 'container']):
        return 'other'
    else:
        return 'other'

# ============================================================================
# Main Detection Pipeline
# ============================================================================

def detect_food_items_with_yolo(img_bytes: bytes, user_pantry: Optional[List] = None) -> List[Dict]:
    """
    Simple YOLOv8-only detection pipeline:
    Image -> Preprocessing -> YOLO Detection -> Item Mapping -> Results
    
    Args:
        img_bytes: Raw image bytes
        user_pantry: Optional user's current pantry (not used in simple version)
    
    Returns:
        List of detected food items with metadata
    """
    global _yolo_model
    
    # Validate input
    if not img_bytes:
        return []
    
    if not isinstance(img_bytes, (bytes, bytearray)):
        return []
    
    # Load YOLO model if needed
    if not _yolo_model_loaded:
        if not load_yolo_model():
            return []
    
    # STEP 0: Image Quality Scoring
    try:
        temp_img = Image.open(io.BytesIO(img_bytes))
        quality_score, quality_metrics = compute_image_quality_score(temp_img)
        
        if quality_score < 0.10:
            print(f" Image quality too low ({quality_score:.2f}), rejecting scan")
            return []
        
        if quality_score < 0.4:
            print(f" Warning: Poor image quality ({quality_score:.2f})")
    except Exception:
        quality_score = 0.8
    
    # STEP 1: Preprocess image
    try:
        img = preprocess_image(img_bytes)
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Image preprocessing failed: {e}")
        return []
    
    # STEP 2: YOLO Object Detection
    detections = detect_objects_with_yolo(img)
    
    if not detections:
        print(" No objects detected by YOLO")
        return []
    
    print(f" YOLO detected {len(detections)} objects")
    
    # STEP 3: Map YOLO classes to food items
    items = []
    item_confidence = {}
    
    for detection in detections:
        try:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Map YOLO class to food item
            food_item = map_yolo_class_to_food_item(class_name)
            
            # Skip if not a food item
            if food_item is None:
                continue
            
            # Create item
            key = food_item.lower().strip()
            
            # Keep highest confidence detection per item
            if key not in item_confidence or confidence > item_confidence[key]:
                category = categorize_item(food_item)
                
                items.append({
                    "name": food_item,
                    "quantity": "1",
                    "expirationDate": None,
                    "category": category,
                    "confidence": confidence,
                    "detection_method": "yolov8",
                    "needs_confirmation": confidence < 0.5,  # Low confidence needs confirmation
                    "bbox": bbox,
                    "quality_score": quality_score,
                    "yolo_class": class_name
                })
                item_confidence[key] = confidence
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Warning: Error processing detection: {e}")
            continue
    
    # STEP 4: Apply quality score weighting
    for item in items:
        base_conf = item.get("confidence", 0)
        quality_weighted = base_conf * quality_score
        blended_conf = (quality_weighted * 0.7) + (base_conf * 0.3)
        item["confidence"] = max(0.0, min(1.0, blended_conf))
    
    # STEP 5: Sort by confidence
    result = sorted(items, key=lambda x: x.get("confidence", 0), reverse=True)
    
    print(f" YOLO Detection Summary: Found {len(result)} food items")
    if len(result) > 0:
        print(f"   Items: {[item.get('name', 'unknown') for item in result[:5]]}")
        confidences = [f"{item.get('confidence', 0):.2f}" for item in result[:5]]
        print(f"   Confidences: {confidences}")
    
    return result

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python pantry_detection_algorithm_yolo_only.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Read image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    # Detect items
    print(f" Analyzing image: {image_path}")
    items = detect_food_items_with_yolo(img_bytes)
    
    # Print results
    print(f"\n Detection complete: {len(items)} items found")
    for i, item in enumerate(items, 1):
        print(f"\n{i}. {item.get('name', 'Unknown')}")
        print(f"   Confidence: {item.get('confidence', 0):.2%}")
        print(f"   Category: {item.get('category', 'unknown')}")
        print(f"   YOLO Class: {item.get('yolo_class', 'unknown')}")
        if item.get('needs_confirmation'):
            print(f"    Needs confirmation")

if __name__ == "__main__":
    main()
