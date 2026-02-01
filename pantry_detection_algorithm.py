import os
import sys
import io
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Environment variables (can be set via os.environ)
ML_VISION_ENABLED = os.getenv("ML_VISION_ENABLED", "true").lower() == "true"
YOLO_DETECTION_ENABLED = os.getenv("YOLO_DETECTION_ENABLED", "true").lower() == "true"
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "s").lower()  # n, s, m, l, x
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

# ============================================================================
# Global Model Variables
# ============================================================================

_ml_models_loaded = False
_yolo_model = None
_clip_model = None
_clip_processor = None
_ocr_reader = None
_food_classifier = None

# Performance optimization: Caching
_clip_cache = {}
_cache_max_size = 200

# ============================================================================
# Model Loading
# ============================================================================

def load_ml_models():
    global _ml_models_loaded, _yolo_model, _clip_model, _clip_processor, _ocr_reader, _food_classifier
    
    if _ml_models_loaded:
        return True
    
    import threading
    
    print("🔄 Loading ML models for photo analysis...")
    
    # Load YOLOv8 for object detection
    if YOLO_DETECTION_ENABLED:
        try:
            from ultralytics import YOLO
            model_name = f"yolov8{YOLO_MODEL_SIZE}.pt"
            _yolo_model = YOLO(model_name)
            print(f"✅ YOLOv8 model loaded ({model_name})")
        except Exception as e:
            print(f"⚠️  Warning: YOLOv8 not available: {e}")
            _yolo_model = None
    
    # Load CLIP for semantic matching
    if ML_VISION_ENABLED:
        try:
            from transformers import CLIPProcessor, CLIPModel
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
            print("✅ CLIP model loaded")
        except Exception as e:
            print(f"⚠️  Warning: CLIP not available: {e}")
            _clip_model = None
            _clip_processor = None
    
    # Load EasyOCR for text recognition
    if ML_VISION_ENABLED:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("✅ EasyOCR reader loaded")
        except Exception as e:
            print(f"⚠️  Warning: EasyOCR not available: {e}")
            _ocr_reader = None
    
    _ml_models_loaded = True
    return True

# ============================================================================
# Image Preprocessing
# ============================================================================

def preprocess_image_for_ml(img_bytes: bytes, angle: Optional[int] = None) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Apply rotation if specified
    if angle is not None and angle in [90, 180, 270]:
        img = img.rotate(-angle, expand=True)
    
    # Resize for performance (800px max dimension)
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    
    # Enhance image quality
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.2)
    
    # Unsharp mask for better edge detection
    try:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    except Exception:
        pass
    
    return img

def compute_image_quality_score(img: Image.Image) -> Tuple[float, Dict]:
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
        
        # Glare score (simplified)
        glare_score = 1.0  # Placeholder
        metrics['glare_score'] = glare_score
        
        # Combined quality score
        quality_score = (blur_score * 0.4 + brightness_score * 0.3 + 
                        contrast_score * 0.2 + glare_score * 0.1)
        
        return quality_score, metrics
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Image quality scoring failed: {e}")
        return 0.8, {}

# ============================================================================
# YOLO Object Detection
# ============================================================================

def stage_yolo_detect_regions(img: Image.Image) -> List[Dict]:
    global _yolo_model
    
    regions = []
    
    if not _yolo_model or not YOLO_DETECTION_ENABLED:
        return regions
    
    try:
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]
        image_area = img_height * img_width
        min_area = 0.01 * image_area  # Minimum 1% of image area
        
        # YOLO detection with optimized thresholds for pantry scenes
        yolo_conf_threshold = 0.05  # Low threshold for pantry scenes
        yolo_iou_threshold = 0.50
        
        # COCO food-like classes (for food prior calculation)
        FOOD_LIKE_CLASSES = {
            'bottle', 'cup', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'fork', 'knife', 'spoon'
        }
        
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
                        
                        # Calculate food prior (0.3 for food-like classes, 0.05 otherwise)
                        yolo_food_prior = 0.3 if class_name.lower() in FOOD_LIKE_CLASSES else 0.05
                        
                        # Determine if container
                        is_container = any(x in class_name.lower() for x in ['bottle', 'jar', 'can', 'box', 'bowl', 'cup', 'vase'])
                        
                        regions.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'yolo_conf': conf,
                            'class_id': class_id,
                            'class_name': class_name,
                            'yolo_food_prior': yolo_food_prior,
                            'is_container': is_container
                        })
                    except Exception as e:
                        if VERBOSE_LOGGING:
                            print(f"Warning: Error processing YOLO box: {e}")
                        continue
        
        return regions
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: YOLO detection failed: {e}")
        return []

def stage_yolo_detect_regions_multi_crop(img: Image.Image, use_multi_crop: bool = False) -> List[Dict]:
    if not use_multi_crop:
        return stage_yolo_detect_regions(img)
    
    regions = []
    
    try:
        img_width, img_height = img.size
        
        # Split into 2x2 overlapping tiles
        tile_width = img_width // 2
        tile_height = img_height // 2
        overlap = 0.3
        
        tiles = []
        for i in range(2):
            for j in range(2):
                x1 = max(0, int(i * tile_width * (1 - overlap)))
                y1 = max(0, int(j * tile_height * (1 - overlap)))
                x2 = min(img_width, x1 + tile_width + int(tile_width * overlap))
                y2 = min(img_height, y1 + tile_height + int(tile_height * overlap))
                tiles.append((x1, y1, x2, y2))
        
        # Run YOLO on each tile
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            try:
                tile = img.crop((x1, y1, x2, y2))
                tile_regions = stage_yolo_detect_regions(tile)
                
                # Adjust bbox coordinates to full image space
                for region in tile_regions:
                    bbox = region["bbox"]
                    region["bbox"] = [
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    ]
                    region["tile_source"] = tile_idx
                    regions.append(region)
            except Exception:
                continue
        
        # Deduplicate overlapping detections
        if len(regions) > 1:
            unique_regions = []
            for region in regions:
                is_duplicate = False
                bbox1 = region["bbox"]
                for existing in unique_regions:
                    bbox2 = existing["bbox"]
                    # Calculate IoU
                    x1_inter = max(bbox1[0], bbox2[0])
                    y1_inter = max(bbox1[1], bbox2[1])
                    x2_inter = min(bbox1[2], bbox2[2])
                    y2_inter = min(bbox1[3], bbox2[3])
                    
                    if x1_inter < x2_inter and y1_inter < y2_inter:
                        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                        union_area = area1 + area2 - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > 0.5:
                            is_duplicate = True
                            if region.get("yolo_conf", 0) > existing.get("yolo_conf", 0):
                                unique_regions.remove(existing)
                                unique_regions.append(region)
                            break
                
                if not is_duplicate:
                    unique_regions.append(region)
            
            regions = unique_regions
        
        return regions
    except Exception:
        return stage_yolo_detect_regions(img)

# ============================================================================
# CLIP Semantic Matching
# ============================================================================

def _get_image_hash(img: Image.Image) -> Optional[str]:
    """Generate hash for image caching"""
    try:
        img_bytes = img.tobytes()
        return hashlib.md5(img_bytes).hexdigest()[:16]
    except Exception:
        return None

def _get_labels_hash(labels: List[str]) -> Optional[str]:
    """Generate hash for labels list"""
    try:
        labels_str = ','.join(sorted([str(l).lower() for l in labels]))
        return hashlib.md5(labels_str.encode()).hexdigest()[:16]
    except Exception:
        return None

def clip_match_open_vocabulary(img_crop: Image.Image, labels: List[str], 
                              use_prompt_engineering: bool = True) -> Optional[Dict]:
    global _clip_model, _clip_processor, _clip_cache
    
    if not _clip_model or not _clip_processor:
        return None
    
    if not labels:
        return None
    
    try:
        import torch
        
        # Check cache
        img_hash = _get_image_hash(img_crop)
        labels_hash = _get_labels_hash(labels)
        if img_hash and labels_hash:
            cache_key = f"{img_hash}_{labels_hash}_{use_prompt_engineering}"
            if cache_key in _clip_cache:
                if VERBOSE_LOGGING:
                    print(f"   ⚡ CLIP cache hit for {len(labels)} labels")
                return _clip_cache[cache_key].copy()
        
        # Prepare prompts
        if use_prompt_engineering:
            prompts = [f"a photo of {label}" for label in labels]
        else:
            prompts = labels
        
        # Process image and text
        inputs = _clip_processor(text=prompts, images=img_crop, return_tensors="pt", padding=True)
        
        with torch.inference_mode():
            outputs = _clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top predictions
        scores = probs[0].cpu().detach().numpy()
        top_indices = np.argsort(scores)[::-1][:3]
        
        best_idx = top_indices[0]
        second_idx = top_indices[1] if len(top_indices) > 1 else best_idx
        
        raw_best = float(scores[best_idx])
        raw_second = float(scores[second_idx])
        
        # Calibration: Temperature scaling
        # Maps raw CLIP scores (typically 0.0-0.3) to meaningful confidence (0.0-1.0)
        calibrated_best = min(1.0, max(0.0, (raw_best - 0.2) / 0.4))
        calibrated_second = min(1.0, max(0.0, (raw_second - 0.2) / 0.4))
        
        result = {
            "label": labels[best_idx],
            "score": calibrated_best,
            "second_best": calibrated_second,
            "raw_score": raw_best,
            "raw_second_best": raw_second,
            "top3_labels": [labels[i] for i in top_indices[:3]],
            "top3_scores": [float(scores[i]) for i in top_indices[:3]]
        }
        
        # Cache result
        if img_hash and labels_hash:
            cache_key = f"{img_hash}_{labels_hash}_{use_prompt_engineering}"
            if len(_clip_cache) > _cache_max_size:
                # Remove oldest 20% of entries
                keys_to_remove = list(_clip_cache.keys())[:int(_cache_max_size * 0.2)]
                for key in keys_to_remove:
                    del _clip_cache[key]
            _clip_cache[cache_key] = result.copy()
        
        return result
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: CLIP matching failed: {e}")
        return None

def stage_classify_food_type(crop: Image.Image, region_info: Optional[Dict] = None) -> Optional[Dict]:
    FOOD_TYPES = [
        "fresh produce",
        "bottle or jar",
        "carton",
        "can",
        "box",
        "bag",
        "snack",
        "beverage"
    ]
    
    food_type_pred = clip_match_open_vocabulary(crop, FOOD_TYPES, use_prompt_engineering=True)
    if not food_type_pred:
        return None
    
    return {
        "food_type": food_type_pred.get("label", ""),
        "score": food_type_pred.get("score", 0.0)
    }

def stage_clip_suggest_label(crop: Image.Image, candidate_labels: List[str], 
                            region_info: Optional[Dict] = None) -> Optional[Dict]:
    if not candidate_labels:
        return None
    
    # Food-only gating (Precision Rule #8) - only if few candidates
    run_food_gate = len(candidate_labels) < 5
    if run_food_gate:
        FOODNESS_LABELS = ["food", "ingredient", "grocery item", "edible item"]
        NON_FOOD_LABELS = ["tool", "furniture", "clothing", "appliance", "object", "non-food"]
        food_check_labels = FOODNESS_LABELS + NON_FOOD_LABELS
        food_check = clip_match_open_vocabulary(crop, food_check_labels, use_prompt_engineering=False)
        
        if food_check:
            food_label = food_check["label"]
            food_score = food_check["score"] if food_label in FOODNESS_LABELS else 0.0
            non_food_score = food_check["score"] if food_label in NON_FOOD_LABELS else 0.0
            clip_food_score = food_score - non_food_score
            
            if clip_food_score < -0.3:
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ CLIP food gate: likely non-food (food_score={clip_food_score:.2f})")
    
    # Run CLIP matching on candidate labels
    clip_pred = clip_match_open_vocabulary(crop, candidate_labels, use_prompt_engineering=True)
    if not clip_pred:
        return None
    
    best = clip_pred["score"]
    second = clip_pred.get("second_best", 0.0)
    raw_best = clip_pred.get("raw_score", 0.0)
    raw_second = clip_pred.get("raw_second_best", 0.0)
    label = clip_pred.get("label", "")
    top3_labels = clip_pred.get("top3_labels", [label] if label else [])
    
    # Check for ambiguity (Precision Rule #3 enhancement)
    margin = best - second
    ambiguity_threshold = 0.07
    ambiguous_label = None
    
    if len(top3_labels) >= 2 and margin < ambiguity_threshold:
        ambiguous_label = " / ".join(top3_labels[:2])
        if VERBOSE_LOGGING:
            print(f"   🔀 Ambiguous prediction (margin={margin:.3f}) - showing: {ambiguous_label}")
    
    # Margin-based selection (Precision Rule #3)
    margin_threshold = 0.01
    
    if len(candidate_labels) >= 3 and margin < margin_threshold:
        if raw_best > 0.01 and raw_second > 0.01:
            if VERBOSE_LOGGING:
                print(f"   ⚠️ CLIP margin too small: {best:.3f} vs {second:.3f} - needs confirmation")
        final_label = ambiguous_label if ambiguous_label else label
        return {
            "label": final_label,
            "score": best,
            "second_best": second,
            "needs_confirmation": True,
            "margin": margin,
            "is_ambiguous": ambiguous_label is not None,
            "top3_labels": top3_labels
        }
    
    # Visual plausibility checks (Precision Rule #5)
    penalty_factor = 1.0
    if region_info:
        is_container = region_info.get("is_container", False)
        class_name = region_info.get("class_name", "").lower()
        
        # Scene prior penalties
        SCENE_PRIOR = {
            "tie": -0.9, "tennis racket": -0.9, "shoe": -0.8, "sock": -0.8,
            "clothing": -0.8, "furniture": -0.7, "tool": -0.7, "appliance": -0.7,
            "food": +0.3, "ingredient": +0.3, "grocery": +0.3
        }
        
        label_lower = label.lower()
        for non_food_key, penalty in SCENE_PRIOR.items():
            if non_food_key in label_lower and penalty < 0:
                penalty_factor *= (1.0 + penalty)
                break
        
        # Check for implausible matches
        if label == "onion" and is_container and "bottle" in class_name:
            penalty_factor *= 0.7
    
    final_score = best * penalty_factor
    
    # Threshold check (use raw score for thresholding)
    raw_threshold = 0.005
    calibrated_threshold = 0.05
    
    if final_score < calibrated_threshold and raw_best < raw_threshold:
        return None
    
    final_label = ambiguous_label if ambiguous_label else label
    
    return {
        "label": final_label,
        "score": min(final_score, 0.8),
        "second_best": second,
        "needs_confirmation": False if ambiguous_label is None else True,
        "margin": margin,
        "is_ambiguous": ambiguous_label is not None,
        "top3_labels": top3_labels
    }

# ============================================================================
# OCR Text Recognition
# ============================================================================

def stage_ocr_bind_to_region(yolo_bbox: List[float], full_image_ocr_results: List, img_size: Tuple[int, int]) -> List[Dict]:
    if not full_image_ocr_results:
        return []
    
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
    yolo_center_x = (yolo_x1 + yolo_x2) / 2
    yolo_center_y = (yolo_y1 + yolo_y2) / 2
    
    bound_texts = []
    
    for ocr_item in full_image_ocr_results:
        if len(ocr_item) < 3:
            continue
        
        ocr_bbox = ocr_item[0]
        ocr_text = ocr_item[1] if len(ocr_item) > 1 else ""
        ocr_conf = ocr_item[2] if len(ocr_item) > 2 else 0.0
        
        if not ocr_text or ocr_conf < 0.1:
            continue
        
        try:
            ocr_points = np.array(ocr_bbox, dtype=np.float32)
            ocr_x1 = float(np.min(ocr_points[:, 0]))
            ocr_y1 = float(np.min(ocr_points[:, 1]))
            ocr_x2 = float(np.max(ocr_points[:, 0]))
            ocr_y2 = float(np.max(ocr_points[:, 1]))
            
            ocr_center_x = (ocr_x1 + ocr_x2) / 2
            ocr_center_y = (ocr_y1 + ocr_y2) / 2
            
            # Check if OCR center is inside YOLO box
            if (yolo_x1 <= ocr_center_x <= yolo_x2 and yolo_y1 <= ocr_center_y <= yolo_y2):
                bound_texts.append({
                    "text": ocr_text,
                    "confidence": ocr_conf,
                    "bbox": [ocr_x1, ocr_y1, ocr_x2, ocr_y2]
                })
            # Check if YOLO center is inside OCR box
            elif (ocr_x1 <= yolo_center_x <= ocr_x2 and ocr_y1 <= yolo_center_y <= ocr_y2):
                bound_texts.append({
                    "text": ocr_text,
                    "confidence": ocr_conf,
                    "bbox": [ocr_x1, ocr_y1, ocr_x2, ocr_y2]
                })
            # Check for significant overlap (IoU > 0.1)
            else:
                overlap_x1 = max(yolo_x1, ocr_x1)
                overlap_y1 = max(yolo_y1, ocr_y1)
                overlap_x2 = min(yolo_x2, ocr_x2)
                overlap_y2 = min(yolo_y2, ocr_y2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    ocr_area = (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1)
                    yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
                    
                    if overlap_area > 0.1 * ocr_area or overlap_area > 0.05 * yolo_area:
                        bound_texts.append({
                            "text": ocr_text,
                            "confidence": ocr_conf,
                            "bbox": [ocr_x1, ocr_y1, ocr_x2, ocr_y2]
                        })
        except Exception:
            continue
    
    return bound_texts

def stage_ocr_propose_candidates(ocr_text: str, clip_model=None, clip_processor=None, crop: Image.Image = None) -> Optional[Dict]:
    """
    OCR proposes candidates → CLIP verifies.
    Example: OCR sees "Kellogg" → candidate = "cereal" → CLIP confirms cereal
    """
    if not ocr_text or len(ocr_text.strip()) < 2:
        return None
    
    # Brand-to-food mapping
    brand_to_food = {
        "kellogg": "cereal", "general mills": "cereal", "quaker": "cereal",
        "barilla": "pasta", "kraft": "macaroni", "campbell": "soup",
        "heinz": "ketchup", "french's": "mustard", "hellmann's": "mayonnaise",
        "lays": "chips", "oreo": "cookies", "clif": "protein bar"
    }
    
    # Food keywords
    food_keywords = {
        "cereal": ["cereal", "granola", "oatmeal", "breakfast", "flakes"],
        "pasta": ["pasta", "spaghetti", "macaroni", "noodles"],
        "rice": ["rice", "jasmine", "basmati"],
        "chips": ["chips", "potato chips"],
        "crackers": ["crackers", "saltines"],
        "soup": ["soup", "broth", "stock"]
    }
    
    ocr_lower = ocr_text.lower()
    proposed_candidates = []
    
    # Check for brand names
    for brand, food_type in brand_to_food.items():
        if brand in ocr_lower:
            proposed_candidates.append(food_type)
    
    # Check for food keywords
    for food_type, keywords in food_keywords.items():
        if any(kw in ocr_lower for kw in keywords):
            if food_type not in proposed_candidates:
                proposed_candidates.append(food_type)
    
    # If we have candidates and CLIP model, verify with CLIP
    if proposed_candidates and clip_model and clip_processor and crop:
        try:
            clip_result = clip_match_open_vocabulary(crop, proposed_candidates, use_prompt_engineering=True)
            if clip_result and clip_result.get("score", 0) > 0.25:
                return {
                    "label": clip_result.get("label", proposed_candidates[0]),
                    "confidence": clip_result.get("score", 0.0),
                    "text": ocr_text,
                    "method": "ocr_clip_verified"
                }
        except Exception:
            pass
    
    # If no CLIP verification, return top candidate with lower confidence
    if proposed_candidates:
        return {
            "label": proposed_candidates[0],
            "confidence": 0.4,
            "text": ocr_text,
            "method": "ocr_proposed"
        }
    
    return None

def stage_ocr_read_label(crop: Image.Image, is_container: bool = False, bound_ocr_texts: Optional[List[Dict]] = None) -> Optional[Dict]:
    
    global _ocr_reader
    
    # If OCR texts are already bound to this region, use them first
    if bound_ocr_texts and len(bound_ocr_texts) > 0:
        combined_text = ' '.join([item["text"] for item in bound_ocr_texts]).lower()
        avg_conf = sum(item.get("confidence", 0.0) for item in bound_ocr_texts) / len(bound_ocr_texts) if bound_ocr_texts else 0.0
        
        if combined_text and len(combined_text) >= 2:
            # Process bound OCR texts through keyword matching
            label_keyword_map = {
                'olive': 'olive oil', 'extra virgin': 'olive oil', 'evoo': 'olive oil',
                'vinegar': 'vinegar', 'balsamic': 'balsamic vinegar',
                'mustard': 'mustard', 'ketchup': 'ketchup', 'mayonnaise': 'mayonnaise',
                'honey': 'honey', 'soy sauce': 'soy sauce',
                'cereal': 'cereal', 'pasta': 'pasta', 'rice': 'rice',
                'soup': 'soup', 'chips': 'chips', 'crackers': 'crackers'
            }
            
            matched_label = None
            matched_confidence = 0.0
            
            for keyword, food_label in label_keyword_map.items():
                if keyword in combined_text:
                    matched_label = food_label
                    matched_confidence = avg_conf
                    break
            
            if matched_label:
                return {
                    "label": matched_label,
                    "confidence": matched_confidence,
                    "text": combined_text
                }
    
    if not _ocr_reader and not bound_ocr_texts:
        return None
    
    if not is_container and not bound_ocr_texts:
        return None
    
    try:
        img_array = np.array(crop)
        ocr_results = _ocr_reader.readtext(img_array)
        
        if not ocr_results:
            return None
        
        # Combine all text with confidence weighting
        texts = []
        confidences = []
        
        for (bbox, text, conf) in ocr_results:
            if conf > 0.3:
                texts.append(text)
                confidences.append(conf)
        
        if not texts:
            return None
        
        # Average confidence
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        combined_text = " ".join(texts).lower()
        
        # Keyword matching
        label_keyword_map = {
            'olive': 'olive oil', 'extra virgin': 'olive oil', 'evoo': 'olive oil',
            'vinegar': 'vinegar', 'balsamic': 'balsamic vinegar',
            'mustard': 'mustard', 'ketchup': 'ketchup', 'mayonnaise': 'mayonnaise',
            'honey': 'honey', 'soy sauce': 'soy sauce',
            'cereal': 'cereal', 'pasta': 'pasta', 'rice': 'rice',
            'soup': 'soup', 'chips': 'chips', 'crackers': 'crackers'
        }
        
        matched_label = None
        for keyword, food_label in label_keyword_map.items():
            if keyword in combined_text:
                matched_label = food_label
                break
        
        return {
            "label": matched_label if matched_label else combined_text,
            "confidence": avg_conf,
            "text": combined_text
        }
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: OCR failed: {e}")
        return None

def extract_expiration_dates(img: Image.Image, ocr_reader) -> List[str]:
    if not ocr_reader:
        return []
    try:
        import re
        img_array = np.array(img)
        ocr_results = ocr_reader.readtext(img_array)
        date_patterns = [
            r'EXP[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'USE\s+BY[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'BEST\s+BY[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        found_dates = []
        for (_bbox, text, confidence) in ocr_results:
            if confidence < 0.5:
                continue
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1) if match.groups() else match.group(0)
                    normalized = normalize_expiration_date(date_str)
                    if normalized:
                        found_dates.append(normalized)
                        break
        return found_dates
    except Exception:
        return []

def normalize_expiration_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    
    try:
        from datetime import datetime
        import re
        
        date_str = date_str.strip()
        
        # Try parsing common formats
        formats = [
            "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d",
            "%m/%d/%y", "%d/%m/%y", "%y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    except Exception:
        return None

def normalize_item_name(name: str) -> Optional[str]:
    if not name:
        return None
    
    name = name.strip().lower()
    
    # Common corrections
    corrections = {
        'milk carton': 'milk',
        'chicken meat': 'chicken',
        'tomatoes': 'tomato',
        'apples': 'apple',
        'potatoes': 'potato',
        'bread loaf': 'bread',
        'pasta box': 'pasta',
        'cereal box': 'cereal'
    }
    
    for wrong, correct in corrections.items():
        if wrong in name:
            name = name.replace(wrong, correct)
    
    # Remove packaging words
    prefixes = ['a ', 'an ', 'the ', 'box of ', 'bottle of ', 'can of ']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    suffixes = [' container', ' package', ' box', ' bottle', ' can']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in name.split()) if name else None

def validate_category(item_name: str, category: str) -> str:
    """Auto-correct category based on item name"""
    if not item_name:
        return category or 'other'
    
    name_lower = item_name.lower()
    
    category_map = {
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'egg', 'cream'],
        'produce': ['apple', 'banana', 'orange', 'tomato', 'lettuce', 'carrot', 'onion'],
        'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey'],
        'beverages': ['juice', 'soda', 'water', 'coffee', 'tea'],
        'bakery': ['bread', 'bagel', 'muffin'],
        'canned_goods': ['soup', 'beans', 'tuna', 'corn'],
        'snacks': ['chips', 'crackers', 'cookies', 'nuts'],
        'condiments': ['ketchup', 'mustard', 'mayonnaise', 'sauce', 'oil', 'vinegar'],
        'grains': ['rice', 'pasta', 'cereal', 'flour', 'oats']
    }
    
    for cat, keywords in category_map.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    
    return category or 'other'

# ============================================================================
# NMS and Deduplication
# ============================================================================

def apply_nms(items: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    if not items or len(items) < 2:
        return items
    
    items_with_bbox = [item for item in items if 'bbox' in item and item.get('bbox')]
    items_without_bbox = [item for item in items if 'bbox' not in item or not item.get('bbox')]
    
    if not items_with_bbox:
        return items
    
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Sort by confidence (descending)
    items_with_bbox.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Apply NMS
    keep = []
    while items_with_bbox:
        current = items_with_bbox.pop(0)
        if current:
            keep.append(current)
        
        # Remove items that overlap significantly with current
        items_with_bbox = [
            item for item in items_with_bbox
            if calculate_iou(current['bbox'], item['bbox']) < iou_threshold
        ]
    
    return keep + items_without_bbox

def apply_ensemble_boosting(result: List[Dict], all_detections: List[Dict]) -> List[Dict]:
    if not all_detections or len(all_detections) < 2:
        return result
    
    # Count detection methods per item
    detection_counts = {}
    for item in all_detections:
        key = item.get("name", "").lower().strip()
        if not key:
            continue
        
        if key not in detection_counts:
            detection_counts[key] = {
                'count': 0,
                'methods': set(),
                'max_confidence': item.get('confidence', 0),
                'item': item
            }
        
        detection_counts[key]['count'] += 1
        method = item.get('detection_method', 'unknown')
        if method:
            detection_counts[key]['methods'].add(method)
        detection_counts[key]['max_confidence'] = max(
            detection_counts[key]['max_confidence'],
            item.get('confidence', 0)
        )
    
    # Boost confidence for items detected by multiple methods
    for item in result:
        key = item.get("name", "").lower().strip()
        if key in detection_counts:
            detection_info = detection_counts[key]
            method_count = len(detection_info['methods'])
            detection_count = detection_info['count']
            
            # Boost for multiple detection methods
            method_boost = 1.0
            if method_count > 1:
                method_boost = 1.0 + (method_count - 1) * 0.10
            
            # Boost for multiple detections of same item
            count_boost = 1.0
            if detection_count > 1:
                count_boost = 1.0 + min(0.20, (detection_count - 1) * 0.05)
            
            # Apply boosts
            boosted_conf = item.get('confidence', 0) * method_boost * count_boost
            item['confidence'] = min(0.85, boosted_conf)
            item['ensemble_boost'] = True
            item['detection_methods'] = list(detection_info['methods'])
            item['detection_count'] = detection_count
    
    return result

# ============================================================================
# Confidence Calibration
# ============================================================================

def calibrate_confidence(items: List[Dict]) -> List[Dict]:
    for item in items:
        method = item.get('detection_method', 'unknown')
        original_conf = item.get('confidence', 0.5)
        name = item.get('name', '').lower()
        
        # Boost factors based on detection method
        boost_factors = {
            'yolov8': 1.15,
            'yolov8+classification': 1.25,
            'ensemble': 1.25,
            'classification': 1.10,
            'unknown': 1.05
        }
        
        factor = boost_factors.get(method, 1.10)
        
        # Additional boosts for specific item characteristics
        additional_boost = 1.0
        
        # Boost for actual food items
        food_keywords = ['apple', 'banana', 'orange', 'bread', 'milk', 'cheese', 'egg',
                        'chicken', 'beef', 'pasta', 'rice', 'cereal', 'soup']
        if any(kw in name for kw in food_keywords):
            additional_boost *= 1.15
        
        # Boost for items detected by YOLO (if confidence is decent)
        if original_conf > 0.20:
            additional_boost *= 1.05
        
        # Disagreement penalty (if multiple labels disagree)
        disagreement_penalty = 1.0
        classifier_label = item.get('classifier_label', '').lower() if item.get('classifier_label') else None
        clip_label = item.get('clip_label', '').lower() if item.get('clip_label') else None
        ocr_label = item.get('ocr_label', '').lower() if item.get('ocr_label') else None
        
        labels_to_check = [l for l in [classifier_label, clip_label, ocr_label] if l]
        if len(labels_to_check) >= 2:
            unique_labels = set(labels_to_check)
            if len(unique_labels) > 1:
                disagreement_penalty = 0.80
        
        # Apply calibration
        if method in ['yolov8+clip', 'yolov8+ocr', 'ensemble']:
            calibrated_conf = min(0.9, original_conf * 1.05 * disagreement_penalty)
        else:
            calibrated_conf = min(0.85, original_conf * factor * disagreement_penalty)
        
        # Never force confidence above actual signal strength
        if original_conf < 0.3:
            calibrated_conf = min(calibrated_conf, original_conf * 1.2)
        
        # Store both original and calibrated
        item['original_confidence'] = original_conf
        item['confidence'] = calibrated_conf
        item['calibration_boost'] = factor * additional_boost * disagreement_penalty
        if disagreement_penalty < 1.0:
            item['disagreement_penalty'] = True
    
    return items

# ============================================================================
# Context Fusion
# ============================================================================

def apply_context_fusion(items: List[Dict], user_pantry: Optional[List] = None, img_height: Optional[int] = None) -> List[Dict]:
    if not user_pantry:
        user_pantry = []
    
    existing_items = set()
    if isinstance(user_pantry, list):
        for item in user_pantry:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    existing_items.add(name.lower().strip())
    
    common_items = {"milk", "eggs", "bread", "cheese", "chicken", "yogurt", "butter", "apple", "banana", "tomato"}
    
    for item in items:
        name_lower = item.get("name", "").lower().strip()
        if not name_lower:
            continue
        
        base_confidence = item.get("confidence", 0)
        boosts = []
        
        # Boost if already in pantry
        if name_lower in existing_items:
            item["confidence"] = min(1.0, base_confidence + 0.15)
            boosts.append("already_in_pantry")
        
        # Boost if common item
        elif name_lower in common_items:
            item["confidence"] = min(1.0, base_confidence + 0.1)
            boosts.append("common_item")
        
        # Shelf-aware spatial context (if bounding box available)
        if img_height and img_height > 0 and 'bbox' in item and item.get('bbox'):
            try:
                bbox = item['bbox']
                if len(bbox) >= 4:
                    y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
                    
                    category = item.get('category', 'other').lower()
                    item_name = name_lower
                    
                    # Door shelves (middle Y-range)
                    if 0.3 <= y_center <= 0.7:
                        door_items = {"beverages", "condiments", "juice", "soda", "ketchup", "mustard"}
                        if any(door_item in item_name or door_item in category for door_item in door_items):
                            item["confidence"] = min(1.0, item.get("confidence", base_confidence) + 0.1)
                            boosts.append("door_shelf_match")
                    
                    # Crisper drawer (bottom)
                    if y_center > 0.75:
                        crisper_items = {"produce", "apple", "banana", "orange", "tomato", "lettuce", "carrot"}
                        if any(crisper_item in item_name or crisper_item in category for crisper_item in crisper_items):
                            item["confidence"] = min(1.0, item.get("confidence", base_confidence) + 0.1)
                            boosts.append("crisper_match")
            except Exception:
                pass
        
        if boosts:
            item["context_boost"] = "+".join(boosts)
    
    return items

# ============================================================================
# Rules Validation
# ============================================================================

def stage_rules_validate(region_data: Dict, clip_suggestion: Optional[Dict], 
                        ocr_result: Optional[Dict], user_pantry: Optional[List] = None) -> Dict:
    
    food_score = region_data.get("food_score", 0.0)
    suggested_label = region_data.get("suggested_label", "")
    confidence = region_data.get("confidence", 0.0)
    
    # Rule 1: If label is clearly non-food, penalize
    non_food_in_fridge = ["tennis racket", "tie", "suitcase", "laptop", "car"]
    if suggested_label.lower() in non_food_in_fridge:
        confidence *= 0.5
        region_data["confidence"] = confidence
        region_data["rules_penalty"] = "non_food_in_context"
    
    # Rule 2: If item is in pantry history, boost
    if user_pantry and isinstance(user_pantry, list):
        user_item_names = [str(item.get('name', '')).lower() for item in user_pantry if isinstance(item, dict)]
        if suggested_label.lower() in user_item_names:
            confidence = min(1.0, confidence * 1.2)
            region_data["confidence"] = confidence
            region_data["rules_boost"] = "in_pantry_history"
    
    # Rule 3: Container handling
    if region_data.get("is_container", False):
        skip_override = region_data.get("skip_container_override", False)
        clip_food_prob = region_data.get("clip_food_prob", 0.0)
        ocr_conf = ocr_result.get("confidence", 0) if ocr_result else 0.0
        
        if skip_override:
            pass  # Keep existing label
        elif ocr_conf >= 0.45:
            pass  # OCR is authoritative
        elif clip_food_prob > 0.0:
            region_data["needs_confirmation"] = True
            if suggested_label not in ["unknown_food", "unknown_packaged_food"]:
                if clip_food_prob < 0.30:
                    region_data["suggested_label"] = f"Possibly: {suggested_label}?"
                confidence = max(0.25, clip_food_prob)
        else:
            region_data["suggested_label"] = "unknown_packaged_food"
            confidence = max(0.2, confidence * 0.6)
            region_data["needs_confirmation"] = True
        
        region_data["confidence"] = confidence
    
    # Rule 4: Context score contribution
    context_score = 0.0
    if user_pantry and isinstance(user_pantry, list):
        user_item_names = [str(item.get('name', '')).lower() for item in user_pantry if isinstance(item, dict)]
        if suggested_label.lower() in user_item_names:
            context_score = 0.3
        else:
            for user_item in user_item_names:
                if suggested_label.lower() in user_item or user_item in suggested_label.lower():
                    context_score = 0.15
                    break
    
    region_data["context_score"] = context_score
    
    # Recalculate food_score with context boost
    if not region_data.get("bypass_rules", False):
        clip_food_prob = region_data.get("clip_food_prob", 0.0)
        if clip_food_prob > 0.35:
            classifier_food_prob = region_data.get("classifier_food_prob", 0.0)
            yolo_food_prior = region_data.get("yolo_food_prior", 0.0)
            food_score = 0.45 * clip_food_prob + 0.25 * classifier_food_prob + 0.20 * yolo_food_prior + 0.10 * context_score
            food_score = max(0.0, min(1.0, food_score))
            region_data["food_score"] = food_score
    
    return region_data

# ============================================================================
# User Decision
# ============================================================================

def stage_user_decision(region_data: Dict) -> Dict:
    """
    STAGE 5: User - "Is this correct?"
    Three-tier confidence policy based on CLIP score.
    - CONFIDENT (≥0.70): Auto-accept
    - LIKELY (≥0.45): Show label + ask confirm
    - UNKNOWN (<0.45): Show label + ask confirm (low confidence)
    """
    clip_score = region_data.get("clip_food_prob", 0.0)
    food_score = region_data.get("food_score", 0.0)
    confidence = region_data.get("confidence", 0.0)
    suggested_label = region_data.get("suggested_label", "")
    
    # Three-tier confidence policy
    if clip_score >= 0.70:
        region_data["auto_add"] = True
        region_data["needs_confirmation"] = False
        region_data["confidence_tier"] = "CONFIDENT"
        region_data["final_confidence"] = min(0.95, max(confidence, clip_score))
    elif clip_score >= 0.45:
        region_data["auto_add"] = False
        region_data["needs_confirmation"] = True
        region_data["confidence_tier"] = "LIKELY"
        region_data["final_confidence"] = min(0.85, max(confidence, clip_score))
        if suggested_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
            region_data["suggested_label"] = suggested_label
    else:
        region_data["auto_add"] = False
        region_data["needs_confirmation"] = True
        region_data["confidence_tier"] = "UNKNOWN"
        region_data["hide"] = False
        
        if suggested_label not in ["unknown_food", "unknown_packaged_food", "unknown_item"]:
            if clip_score < 0.40:
                region_data["suggested_label"] = f"{suggested_label} (?)"
            else:
                region_data["suggested_label"] = suggested_label
            region_data["final_confidence"] = max(0.2, min(0.6, clip_score * 1.2))
        else:
            region_data["suggested_label"] = "unknown_food (?)"
            region_data["final_confidence"] = max(0.1, clip_score)
    
    return region_data

# ============================================================================
# Context-Aware Candidate Lists
# ============================================================================

def route_candidates(food_type: Optional[str] = None, region_type: Optional[str] = None, 
                    yolo_class: str = "", ocr_text: str = "", 
                    user_pantry_items: Optional[List] = None) -> List[str]:
    
    candidates = []
    yolo_lower = yolo_class.lower()
    ocr_lower = ocr_text.lower() if ocr_text else ""
    
    # Add user's past items FIRST (highest priority)
    user_items = []
    if user_pantry_items and isinstance(user_pantry_items, list):
        for item in user_pantry_items:
            if isinstance(item, dict):
                name = item.get("name", "").strip()
                if name and name.lower() not in ["unknown_food", "unknown_packaged_food"]:
                    user_items.append(name)
    
    if user_items:
        candidates.extend(user_items[:5])
    
    # Labels by food type (narrowed answer space)
    LABELS_BY_TYPE = {
        "fresh produce": [
            "apple", "banana", "orange", "onion", "garlic", "tomato", "potato", "carrot",
            "lettuce", "spinach", "broccoli", "pepper", "cucumber"
        ],
        "bottle or jar": [
            "olive oil", "vegetable oil", "soy sauce", "vinegar", "honey", "mustard",
            "ketchup", "mayonnaise", "juice", "water"
        ],
        "carton": [
            "milk", "juice", "cream", "yogurt", "eggs"
        ],
        "can": [
            "soda", "cola", "beans", "soup", "tomato sauce", "tuna"
        ],
        "box": [
            "cereal", "pasta", "rice", "crackers", "cookies", "flour", "sugar"
        ],
        "bag": [
            "chips", "nuts", "flour", "sugar", "rice", "bread"
        ],
        "snack": [
            "chips", "cookies", "crackers", "granola bar", "protein bar", "nuts"
        ],
        "beverage": [
            "water", "juice", "soda", "cola", "milk", "coffee", "tea"
        ]
    }
    
    # Use food_type to narrow candidates
    if food_type and food_type in LABELS_BY_TYPE:
        type_labels = LABELS_BY_TYPE[food_type]
        candidates.extend(type_labels)
    else:
        # Fallback to region_type heuristics
        if region_type == "container":
            candidates.extend(LABELS_BY_TYPE.get("bottle or jar", []))
        elif region_type == "produce":
            candidates.extend(LABELS_BY_TYPE.get("fresh produce", []))
        else:
            candidates.extend(LABELS_BY_TYPE.get("box", []))
    
    # OCR hints
    ocr_hints = []
    if ocr_lower:
        if any(kw in ocr_lower for kw in ["cereal", "granola", "oats"]):
            ocr_hints.extend(["cereal", "oats", "granola bar"])
        if any(kw in ocr_lower for kw in ["pasta", "spaghetti", "macaroni"]):
            ocr_hints.extend(["pasta", "spaghetti", "macaroni"])
        if any(kw in ocr_lower for kw in ["rice", "jasmine", "basmati"]):
            ocr_hints.extend(["rice"])
    
    if ocr_hints:
        candidates = ocr_hints + [c for c in candidates if c not in ocr_hints]
    
    # Hard constraints - limit to 20 candidates
    if len(candidates) > 20:
        candidates = candidates[:20]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for item in candidates:
        item_lower = item.lower()
        if item_lower not in seen:
            seen.add(item_lower)
            unique_candidates.append(item)
    
    return unique_candidates

def get_context_aware_candidates(region_info: Dict, food_type: Optional[str] = None, 
                                ocr_text: str = "", user_pantry_items: Optional[List] = None) -> List[str]:
   
    return route_candidates(
        food_type=food_type,
        region_type="container" if region_info.get("is_container", False) else "produce",
        yolo_class=region_info.get("class_name", ""),
        ocr_text=ocr_text,
        user_pantry_items=user_pantry_items
    )

# ============================================================================
# Main Detection Pipeline
# ============================================================================

def detect_food_items_with_ml(img_bytes: bytes, user_pantry: Optional[List] = None, 
                              skip_preprocessing: bool = False, use_multi_angle: bool = True) -> List[Dict]:
    
    global _yolo_model, _clip_model, _ocr_reader, _ml_models_loaded
    
    # Validate input
    if not img_bytes:
        return []
    
    if not isinstance(img_bytes, (bytes, bytearray)):
        return []
    
    # Load models if needed
    if not _ml_models_loaded:
        load_ml_models()
    
    # STEP 0: Image Quality Scoring
    try:
        temp_img = Image.open(io.BytesIO(img_bytes)) if not skip_preprocessing else img_bytes
        quality_score, quality_metrics = compute_image_quality_score(temp_img)
        
        if quality_score < 0.10:
            print(f"⚠️ Image quality too low ({quality_score:.2f}), rejecting scan")
            return []
        
        if quality_score < 0.4:
            print(f"⚠️ Warning: Poor image quality ({quality_score:.2f})")
    except Exception:
        quality_score = 0.8
    
    # STEP 1: Preprocess image
    try:
        if skip_preprocessing:
            img = img_bytes
        else:
            img = preprocess_image_for_ml(img_bytes)
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Warning: Image preprocessing failed: {e}")
        return []
    
    items = []
    item_confidence = {}
    
    # Extract expiration dates (global)
    expiration_dates = extract_expiration_dates(img, _ocr_reader)
    exp_date = expiration_dates[0] if expiration_dates else None
    
    # STEP 2: YOLO Object Detection
    regions = stage_yolo_detect_regions(img)
    
    # Multi-crop fallback if no regions found
    if not regions or len(regions) == 0:
        if VERBOSE_LOGGING:
            print("   🔍 YOLO found no regions - trying multi-crop zooming")
        regions = stage_yolo_detect_regions_multi_crop(img, use_multi_crop=True)
    
    # Run full-image OCR once for spatial binding (performance optimization)
    full_image_ocr_results = None
    if _ocr_reader:
        try:
            img_array = np.array(img)
            max_ocr_size = 2000
            if img_array.shape[0] > max_ocr_size or img_array.shape[1] > max_ocr_size:
                scale = min(max_ocr_size / img_array.shape[0], max_ocr_size / img_array.shape[1])
                new_h, new_w = int(img_array.shape[0] * scale), int(img_array.shape[1] * scale)
                img_resized = Image.fromarray(img_array).resize((new_w, new_h), Image.Resampling.LANCZOS)
                img_array_ocr = np.array(img_resized)
                ocr_scale = scale
            else:
                img_array_ocr = img_array
                ocr_scale = 1.0
            
            full_image_ocr_results = _ocr_reader.readtext(img_array_ocr, detail=1, width_ths=0.7, height_ths=0.7)
            
            # Scale bboxes back to original image size if resized
            if ocr_scale != 1.0:
                for ocr_item in full_image_ocr_results:
                    if len(ocr_item) > 0 and isinstance(ocr_item[0], (list, np.ndarray)):
                        ocr_item[0] = (np.array(ocr_item[0]) / ocr_scale).tolist()
            
            if VERBOSE_LOGGING and full_image_ocr_results:
                print(f"   📍 OCR spatial binding: Found {len(full_image_ocr_results)} text regions")
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"   ⚠️ Full-image OCR error: {e}")
    
    if regions:
        for region in regions:
            try:
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                
                # Crop region
                crop = img.crop((x1, y1, x2, y2))
                
                class_name = region.get('class_name', '')
                is_container = region.get('is_container', False)
                yolo_conf = region.get('yolo_conf', 0.5)
                yolo_food_prior = region.get('yolo_food_prior', 0.05)
                
                region_info = {
                    'bbox': bbox,
                    'class_name': class_name,
                    'is_container': is_container,
                    'confidence': yolo_conf,
                    'yolo_conf': yolo_conf,
                    'yolo_food_prior': yolo_food_prior
                }
                
                # STAGE 1.5: Food Type Classification
                food_type_result = stage_classify_food_type(crop, region_info)
                food_type = food_type_result.get("food_type") if food_type_result else None
                
                # STAGE 3: OCR Spatial Binding
                bound_ocr_texts = []
                if full_image_ocr_results:
                    bound_ocr_texts = stage_ocr_bind_to_region(bbox, full_image_ocr_results, img.size)
                
                # STAGE 3: OCR Read Label (with spatial binding)
                ocr_result = None
                ocr_authoritative = False
                
                if is_container or bound_ocr_texts:
                    ocr_result = stage_ocr_read_label(crop, is_container=True, bound_ocr_texts=bound_ocr_texts)
                    
                    # OCR candidate proposal if OCR didn't find label
                    if (not ocr_result or not ocr_result.get("label")) and bound_ocr_texts:
                        combined_ocr_text = ' '.join([t["text"] for t in bound_ocr_texts])
                        ocr_candidate_result = stage_ocr_propose_candidates(
                            combined_ocr_text,
                            clip_model=_clip_model,
                            clip_processor=_clip_processor,
                            crop=crop
                        )
                        if ocr_candidate_result:
                            ocr_result = {
                                "label": ocr_candidate_result.get("label"),
                                "confidence": ocr_candidate_result.get("confidence", 0.4),
                                "text": ocr_candidate_result.get("text"),
                                "method": ocr_candidate_result.get("method", "ocr_proposed")
                            }
                    
                    # Check if OCR is authoritative
                    if ocr_result and ocr_result.get("label") and ocr_result.get("confidence", 0) > 0.45:
                        ocr_authoritative = True
                
                ocr_text = ocr_result.get("text", "") if ocr_result else ""
                
                # STAGE 2: Get context-aware candidates (only if OCR not authoritative)
                candidate_labels = []
                if not ocr_authoritative:
                    candidate_labels = get_context_aware_candidates(
                        region_info,
                        food_type=food_type,
                        ocr_text=ocr_text,
                        user_pantry_items=user_pantry
                    )
                
                # STAGE 2: CLIP Classification (with multi-crop voting)
                clip_suggestion = None
                clip_predictions = []
                
                if not ocr_authoritative and candidate_labels:
                    # Multi-crop voting (Precision Rule #4)
                    crops_to_classify = []
                    
                    # Tight crop
                    crops_to_classify.append(("tight", crop))
                    
                    # Expanded crop (+20% padding)
                    crop_width = int(x2) - int(x1)
                    crop_height = int(y2) - int(y1)
                    padding_x = int(crop_width * 0.2)
                    padding_y = int(crop_height * 0.2)
                    expanded_x1 = max(0, int(x1) - padding_x)
                    expanded_y1 = max(0, int(y1) - padding_y)
                    expanded_x2 = min(img.size[0], int(x2) + padding_x)
                    expanded_y2 = min(img.size[1], int(y2) + padding_y)
                    expanded_crop = img.crop((expanded_x1, expanded_y1, expanded_x2, expanded_y2))
                    crops_to_classify.append(("expanded", expanded_crop))
                    
                    # Run CLIP on all crops
                    for crop_name, crop_img in crops_to_classify:
                        clip_pred = stage_clip_suggest_label(crop_img, candidate_labels, region_info=region_info)
                        if clip_pred:
                            clip_predictions.append((crop_name, clip_pred))
                    
                    # Weighted voting
                    if clip_predictions:
                        from collections import Counter
                        crop_weights = {"tight": 0.5, "expanded": 0.3}
                        weighted_scores = {}
                        total_weight = 0.0
                        vote_counts = Counter()
                        
                        for crop_name, pred in clip_predictions:
                            label = pred.get("label")
                            score = pred.get("score", 0)
                            if label:
                                weight = crop_weights.get(crop_name, 0.3) * score
                                weighted_scores[label] = weighted_scores.get(label, 0.0) + weight
                                total_weight += weight
                                vote_counts[label] += 1
                        
                        if weighted_scores:
                            best_label = max(weighted_scores, key=weighted_scores.get)
                            winning_scores = [p.get("score", 0) for c, p in clip_predictions if p.get("label") == best_label]
                            avg_score = sum(winning_scores) / len(winning_scores) if winning_scores else 0.0
                            
                            clip_suggestion = {
                                "label": best_label,
                                "score": avg_score,
                                "second_best": max([p.get("second_best", 0) for c, p in clip_predictions]),
                                "needs_confirmation": any(p.get("needs_confirmation", False) for c, p in clip_predictions),
                                "votes": vote_counts[best_label],
                                "total_crops": len(crops_to_classify)
                            }
                
                # OCR boost (if OCR text matches CLIP label)
                ocr_boost_applied = False
                if not ocr_authoritative and ocr_result and ocr_result.get("text") and clip_suggestion:
                    ocr_text_lower = ocr_result.get("text", "").lower()
                    clip_label_lower = clip_suggestion.get("label", "").lower()
                    if clip_label_lower in ocr_text_lower or ocr_text_lower in clip_label_lower:
                        original_clip_score = clip_suggestion.get("score", 0.0)
                        boosted_score = min(1.0, original_clip_score + 0.25)
                        clip_suggestion["score"] = boosted_score
                        ocr_boost_applied = True
                
                # DECISION LADDER (Precision Rule #10)
                suggested_label = None
                clip_food_prob = clip_suggestion.get("score", 0.0) if clip_suggestion else 0.0
                ocr_confidence = ocr_result.get("confidence", 0.0) if ocr_result else 0.0
                needs_confirmation_flag = False
                detection_method = "yolov8"
                
                # Rule: OCR strong (≥0.45) > CLIP very strong (≥0.45) > CLIP moderate > CLIP weak
                if ocr_authoritative or (ocr_result and ocr_result.get("confidence", 0) >= 0.45):
                    suggested_label = ocr_result["label"]
                    ocr_confidence = ocr_result.get("confidence", 0.9 if ocr_authoritative else ocr_result.get("confidence", 0))
                    detection_method = "yolov8+ocr"
                elif clip_suggestion and clip_food_prob >= 0.45:
                    suggested_label = clip_suggestion["label"]
                    detection_method = "yolov8+clip"
                    needs_confirmation_flag = False
                elif clip_suggestion:
                    suggested_label = clip_suggestion["label"]
                    detection_method = "yolov8+clip"
                    needs_confirmation_flag = True
                else:
                    suggested_label = "unknown_food"
                    needs_confirmation_flag = True
                
                # Food score calculation
                food_score = 0.0
                if ocr_confidence > 0.45:
                    food_score = 0.6 * ocr_confidence + 0.2 * clip_food_prob + 0.1 * yolo_food_prior + 0.1 * (yolo_conf * 0.5)
                elif clip_food_prob > 0.35:
                    food_score = 0.45 * clip_food_prob + 0.20 * yolo_food_prior + 0.10 * 0.0  # classifier_food_prob would go here
                else:
                    food_score = 0.3 * clip_food_prob + 0.2 * ocr_confidence + 0.3 * yolo_food_prior + 0.2 * (yolo_conf * 0.5)
                
                food_score = max(0.0, min(1.0, food_score))
                confidence_direct = max(clip_food_prob, ocr_confidence, yolo_conf * 0.5)
                
                # Prepare region data for rules stage
                region_data = {
                    "bbox": bbox,
                    "yolo_conf": yolo_conf,
                    "yolo_food_prior": yolo_food_prior,
                    "is_container": is_container,
                    "suggested_label": suggested_label or "unknown_food",
                    "clip_suggestion": clip_suggestion,
                    "ocr_result": ocr_result,
                    "food_score": food_score,
                    "confidence": confidence_direct,
                    "classifier_food_prob": 0.0,  # Would be set if food classifier available
                    "clip_food_prob": clip_food_prob,
                    "detection_method": detection_method,
                    "needs_confirmation": needs_confirmation_flag,
                    "skip_container_override": ocr_authoritative or (clip_food_prob >= 0.45),
                    "bypass_rules": ocr_authoritative or (clip_food_prob >= 0.45)
                }
                
                # STAGE 4: Rules Validation
                if not region_data.get("bypass_rules", False):
                    region_data = stage_rules_validate(region_data, clip_suggestion, ocr_result, user_pantry)
                
                # STAGE 5: User Decision
                region_data = stage_user_decision(region_data)
                
                # Build final item
                final_label = region_data["suggested_label"]
                final_conf = region_data.get("final_confidence", region_data["confidence"])
                
                # Extract expiration date if OCR found one
                exp_date_item = None
                if ocr_result and ocr_result.get("text"):
                    try:
                        exp_dates = extract_expiration_dates(crop, _ocr_reader)
                        if exp_dates:
                            exp_date_item = exp_dates[0]
                    except Exception:
                        pass
                
                # Confidence bands
                if final_conf > 0.8:
                    confidence_band = "high"
                    needs_confirmation = False
                elif final_conf >= 0.5:
                    confidence_band = "medium"
                    needs_confirmation = True
                else:
                    confidence_band = "low"
                    needs_confirmation = True
                
                # Create item
                key = final_label.lower().strip()
                if key not in item_confidence or final_conf > item_confidence[key]:
                    normalized_name = normalize_item_name(final_label)
                    category = validate_category(normalized_name or final_label, "other")
                    
                    items.append({
                        "name": normalized_name or final_label,
                        "quantity": "1",
                        "expirationDate": exp_date_item or exp_date,
                        "category": category,
                        "confidence": final_conf,
                        "detection_method": detection_method,
                        "needs_confirmation": needs_confirmation,
                        "bbox": bbox,
                        "quality_score": quality_score,
                        "confidence_band": confidence_band,
                        "clip_label": clip_suggestion.get("label") if clip_suggestion else None,
                        "ocr_label": ocr_result.get("label") if ocr_result else None
                    })
                    item_confidence[key] = final_conf
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Warning: Error processing region: {e}")
                continue
    
    # STEP 4: Apply NMS
    items = apply_nms(items, iou_threshold=0.5)
    
    # STEP 5: Apply ensemble boosting
    items = apply_ensemble_boosting(items, items)
    
    # STEP 6: Apply context fusion
    if user_pantry:
        img_height = img.size[1] if hasattr(img, 'size') else None
        items = apply_context_fusion(items, user_pantry, img_height=img_height)
    
    # STEP 7: Apply quality score weighting
    for item in items:
        base_conf = item.get("confidence", 0)
        quality_weighted = base_conf * quality_score
        blended_conf = (quality_weighted * 0.7) + (base_conf * 0.3)
        min_quality_multiplier = 0.3
        final_conf = max(blended_conf, base_conf * min_quality_multiplier)
        item["confidence"] = max(0.0, min(1.0, final_conf))
        item["quality_score"] = quality_score
    
    # STEP 8: Calibrate confidence
    items = calibrate_confidence(items)
    
    # STEP 9: Sort by confidence
    result = sorted(items, key=lambda x: x.get("confidence", 0), reverse=True)
    
    print(f"📊 ML Detection Summary: Found {len(result)} items")
    if len(result) > 0:
        print(f"   Items: {[item.get('name', 'unknown') for item in result[:5]]}")
        confidences = [f"{item.get('confidence', 0):.2f}" for item in result[:5]]
        print(f"   Confidences: {confidences}")
        print(f"   Detection methods: {[item.get('detection_method', 'unknown') for item in result[:5]]}")
    
    return result

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python pantry_detection_algorithm.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Read image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    # Detect items
    print(f"🔍 Analyzing image: {image_path}")
    items = detect_food_items_with_ml(img_bytes)
    
    # Print results
    print(f"\n✅ Detection complete: {len(items)} items found")
    for i, item in enumerate(items, 1):
        print(f"\n{i}. {item.get('name', 'Unknown')}")
        print(f"   Confidence: {item.get('confidence', 0):.2%}")
        print(f"   Method: {item.get('detection_method', 'unknown')}")
        if item.get('needs_confirmation'):
            print(f"   ⚠️ Needs confirmation")

if __name__ == "__main__":
    main()
