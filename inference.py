import torch
import argparse
import json
import os
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import re
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np


class LicensePlateInference:
    """Inference class for license plate detection using fine-tuned Qwen2.5-VL model."""
    
    def __init__(self, model_path: str, base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to the fine-tuned LoRA model
            base_model_name: Name of the base model
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and processor."""
        print(f"Loading base model: {self.base_model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        # Load base model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights if available
        if os.path.exists(self.model_path):
            print(f"Loading LoRA weights from: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            print(f"Warning: LoRA weights not found at {self.model_path}")
            print("Using base model without fine-tuning")
            
        self.model.eval()
        
    def parse_bounding_boxes(self, text: str) -> List[Tuple[int, int, int, int]]:
        """
        Parse bounding boxes from model output.
        
        Args:
            text: Model output text containing location tokens or JSON format
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        boxes = []
        
        # First try to parse location tokens (original format)
        loc_pattern = r'<loc(\d{4})>'
        matches = re.findall(loc_pattern, text)
        
        # Group matches into boxes (every 4 consecutive matches)
        for i in range(0, len(matches), 4):
            if i + 3 < len(matches):
                x1, y1, x2, y2 = [int(m) for m in matches[i:i+4]]
                boxes.append((x1, y1, x2, y2))
        
        # If no location tokens found, try to parse JSON format
        if not boxes:
            try:
                import json
                # Look for JSON content in the text
                json_pattern = r'```json\s*(.*?)\s*```'
                json_matches = re.findall(json_pattern, text, re.DOTALL)
                
                for json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'bbox_2d' in item:
                                    bbox = item['bbox_2d']
                                    if len(bbox) == 4:
                                        x1, y1, x2, y2 = bbox
                                        boxes.append((x1, y1, x2, y2))
                    except json.JSONDecodeError:
                        continue
            except:
                pass
                
        return boxes
        
    def normalize_coordinates(self, boxes: List[Tuple[int, int, int, int]], 
                            image_width: int, image_height: int, 
                            input_text: str = "") -> List[Tuple[float, float, float, float]]:
        """
        Convert coordinates to actual pixel coordinates.
        
        Args:
            boxes: List of bounding boxes 
            image_width: Actual image width
            image_height: Actual image height
            input_text: Original text to determine format
            
        Returns:
            List of bounding boxes in actual pixel coordinates
        """
        normalized_boxes = []
        
        # Check if we're dealing with JSON format (actual coordinates) or loc tokens (0-1000 scale)
        is_json_format = "```json" in input_text
        
        for x1, y1, x2, y2 in boxes:
            if is_json_format:
                # JSON format already gives actual pixel coordinates
                actual_x1, actual_y1, actual_x2, actual_y2 = float(x1), float(y1), float(x2), float(y2)
            else:
                # Location token format uses 0-1000 scale
                actual_x1 = (x1 / 1000.0) * image_width
                actual_y1 = (y1 / 1000.0) * image_height
                actual_x2 = (x2 / 1000.0) * image_width
                actual_y2 = (y2 / 1000.0) * image_height
            
            normalized_boxes.append((actual_x1, actual_y1, actual_x2, actual_y2))
            
        return normalized_boxes
        
    def predict(self, image_path: str, confidence_threshold: float = 0.5) -> dict:
        """
        Predict license plates in an image.
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence threshold (not used in current implementation)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        
        # Prepare input with proper image token formatting for Qwen2.5-VL
        question = "detect License-Plate"
        # Use the proper conversation format with image token
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Process inputs using apply_chat_template
        text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            
        # Decode output
        full_response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "Answer: " in full_response:
            answer = full_response.split("Answer: ", 1)[1].strip()
        else:
            answer = full_response.strip()
            
        # Parse bounding boxes
        boxes_normalized = self.parse_bounding_boxes(answer)
        boxes_actual = self.normalize_coordinates(boxes_normalized, image_width, image_height, answer)
        
        # Create results
        results = {
            "image_path": image_path,
            "image_size": (image_width, image_height),
            "raw_response": full_response,
            "parsed_answer": answer,
            "num_detections": len(boxes_actual),
            "detections": []
        }
        
        # Add detection details
        for i, ((x1, y1, x2, y2), (norm_x1, norm_y1, norm_x2, norm_y2)) in enumerate(zip(boxes_actual, boxes_normalized)):
            detection = {
                "id": i,
                "class": "License-Plate",
                "confidence": 1.0,  # Model doesn't output confidence scores
                "bbox_actual": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1
                },
                "bbox_normalized": {
                    "x1": norm_x1, "y1": norm_y1, "x2": norm_x2, "y2": norm_y2
                }
            }
            results["detections"].append(detection)
            
        return results
        
    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        Predict license plates for multiple images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "num_detections": 0,
                    "detections": []
                })
                
        return results
        
    def visualize_predictions(self, image_path: str, output_path: str = None):
        """
        Visualize predictions on an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
        """
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            print("PIL not available for visualization")
            return
            
        # Get predictions
        results = self.predict(image_path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw bounding boxes
        for detection in results["detections"]:
            bbox = detection["bbox_actual"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label = f"{detection['class']}"
            draw.text((x1, y1 - 20), label, fill="red")
            
        # Save or show image
        if output_path:
            image.save(output_path)
            print(f"Visualization saved to: {output_path}")
        else:
            image.show()
            
        return image
    
    def visualize_predictions_matplotlib(self, image_path: str, output_path: str = None, 
                                       ground_truth_boxes: List[Tuple[int, int, int, int]] = None):
        """
        Create visualization using matplotlib with both predictions and ground truth.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            ground_truth_boxes: List of ground truth bounding boxes in normalized coordinates
        """
        # Get predictions
        results = self.predict(image_path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw prediction boxes in red
        for detection in results["detections"]:
            bbox = detection["bbox_actual"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            
            # Create rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none', label='Prediction')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1 - 5, f"Pred: {detection['class']}", 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                   fontsize=8, color='white')
        
        # Draw ground truth boxes in green if provided
        if ground_truth_boxes:
            for i, (x1, y1, x2, y2) in enumerate(ground_truth_boxes):
                # Convert normalized coordinates to actual coordinates
                actual_x1 = (x1 / 1000.0) * image_width
                actual_y1 = (y1 / 1000.0) * image_height
                actual_x2 = (x2 / 1000.0) * image_width
                actual_y2 = (y2 / 1000.0) * image_height
                
                # Create rectangle
                rect = Rectangle((actual_x1, actual_y1), actual_x2-actual_x1, actual_y2-actual_y1,
                               linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth' if i == 0 else "")
                ax.add_patch(rect)
                
                # Add label
                ax.text(actual_x1, actual_y1 - 25, f"GT: License-Plate", 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                       fontsize=8, color='white')
        
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_title(f"License Plate Detection Results\\n{os.path.basename(image_path)}", fontsize=14)
        ax.axis('off')
        
        # Add legend
        if ground_truth_boxes:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='red', lw=2, label='Predictions'),
                             Line2D([0], [0], color='green', lw=2, label='Ground Truth')]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        return fig


def load_test_set(test_annotations_path: str = "dataset/_annotations.test.jsonl") -> List[dict]:
    """Load test set annotations from JSONL file."""
    test_data = []
    
    with open(test_annotations_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data.append(data)
    
    return test_data


def parse_ground_truth_boxes(suffix: str) -> List[Tuple[int, int, int, int]]:
    """Parse ground truth bounding boxes from annotation suffix."""
    boxes = []
    
    # Find all location tokens in ground truth
    loc_pattern = r'<loc(\d{4})>'
    matches = re.findall(loc_pattern, suffix)
    
    # Group matches into boxes (every 4 consecutive matches)
    for i in range(0, len(matches), 4):
        if i + 3 < len(matches):
            x1, y1, x2, y2 = [int(m) for m in matches[i:i+4]]
            boxes.append((x1, y1, x2, y2))
    
    return boxes


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def evaluate_predictions(predictions: List[dict], ground_truth: List[dict], iou_threshold: float = 0.5) -> dict:
    """Evaluate predictions against ground truth."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Create mapping from image name to ground truth
    gt_dict = {gt['image']: gt for gt in ground_truth}
    
    for pred in predictions:
        image_name = os.path.basename(pred['image_path'])
        
        if image_name not in gt_dict:
            continue
            
        gt = gt_dict[image_name]
        gt_boxes = parse_ground_truth_boxes(gt['suffix'])
        pred_boxes = [(int(d['bbox_normalized']['x1']), int(d['bbox_normalized']['y1']), 
                      int(d['bbox_normalized']['x2']), int(d['bbox_normalized']['y2'])) 
                     for d in pred['detections']]
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
                true_positives += 1
        
        # Count false positives and false negatives
        false_positives += len(pred_boxes) - len(matched_pred)
        false_negatives += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="License Plate Detection Inference")
    parser.add_argument("--model_path", type=str, default="./lora_model", help="Path to fine-tuned model")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name")
    parser.add_argument("--test_only", action="store_true", help="Only process test set")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = LicensePlateInference(args.model_path, args.base_model)
    
    # Load test set
    test_annotations_path = os.path.join(args.dataset_dir, "_annotations.test.jsonl")
    test_data = load_test_set(test_annotations_path)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Process test images
    test_image_paths = []
    for item in test_data:
        image_path = os.path.join(args.dataset_dir, item['image'])
        if os.path.exists(image_path):
            test_image_paths.append(image_path)
    
    print(f"Processing {len(test_image_paths)} test images")
    results = inference.predict_batch(test_image_paths)
    
    # Save results
    output_file = os.path.join(args.output_dir, "test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved to: {output_file}")
    
    # Evaluate predictions
    evaluation = evaluate_predictions(results, test_data)
    
    # Save evaluation metrics
    eval_file = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(eval_file, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation metrics saved to: {eval_file}")
    
    # Print evaluation summary
    print(f"\nEvaluation Results:")
    print(f"True Positives: {evaluation['true_positives']}")
    print(f"False Positives: {evaluation['false_positives']}")
    print(f"False Negatives: {evaluation['false_negatives']}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1-Score: {evaluation['f1_score']:.4f}")
    
    # Print summary
    total_detections = sum(r['num_detections'] for r in results)
    print(f"\nDetection Summary:")
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(results):.2f}")
    
    # Create visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        print(f"\nCreating visualizations...")
        # Create mapping from image name to ground truth
        gt_dict = {gt['image']: gt for gt in test_data}
        
        for i, result in enumerate(results[:10]):  # Visualize first 10 results
            image_name = os.path.basename(result['image_path'])
            output_path = os.path.join(viz_dir, f"viz_{image_name.replace('.jpg', '.png')}")
            
            # Get ground truth boxes for this image
            ground_truth_boxes = None
            if image_name in gt_dict:
                ground_truth_boxes = parse_ground_truth_boxes(gt_dict[image_name]['suffix'])
            
            inference.visualize_predictions_matplotlib(result['image_path'], output_path, ground_truth_boxes)
        
        print(f"Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()