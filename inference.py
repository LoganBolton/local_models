import torch
import argparse
import json
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import re
from typing import List, Tuple, Optional


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
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
            text: Model output text containing location tokens
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        boxes = []
        
        # Find all location tokens
        loc_pattern = r'<loc(\d{4})>'
        matches = re.findall(loc_pattern, text)
        
        # Group matches into boxes (every 4 consecutive matches)
        for i in range(0, len(matches), 4):
            if i + 3 < len(matches):
                x1, y1, x2, y2 = [int(m) for m in matches[i:i+4]]
                boxes.append((x1, y1, x2, y2))
                
        return boxes
        
    def normalize_coordinates(self, boxes: List[Tuple[int, int, int, int]], 
                            image_width: int, image_height: int) -> List[Tuple[float, float, float, float]]:
        """
        Convert normalized coordinates (0-1000) to actual pixel coordinates.
        
        Args:
            boxes: List of bounding boxes in 0-1000 scale
            image_width: Actual image width
            image_height: Actual image height
            
        Returns:
            List of bounding boxes in actual pixel coordinates
        """
        normalized_boxes = []
        
        for x1, y1, x2, y2 in boxes:
            # Convert from 0-1000 scale to actual coordinates
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
        
        # Prepare input
        question = "detect License-Plate"
        conversation = f"Question: {question}\nAnswer: "
        
        # Process inputs
        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt",
            max_length=512,
            truncation=True,
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
                temperature=0.1,
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
        boxes_actual = self.normalize_coordinates(boxes_normalized, image_width, image_height)
        
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


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="License Plate Detection Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = LicensePlateInference(args.model_path, args.base_model)
    
    # Process images
    if args.image_path:
        # Single image
        print(f"Processing image: {args.image_path}")
        results = inference.predict(args.image_path)
        
        # Print results
        print(f"\nResults for {args.image_path}:")
        print(f"Number of detections: {results['num_detections']}")
        print(f"Raw response: {results['raw_response']}")
        
        for i, detection in enumerate(results['detections']):
            bbox = detection['bbox_actual']
            print(f"Detection {i+1}: [{bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}]")
            
        # Save results
        output_file = os.path.join(args.output_dir, "prediction_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
        
        # Visualize if requested
        if args.visualize:
            output_vis = os.path.join(args.output_dir, "visualization.jpg")
            inference.visualize_predictions(args.image_path, output_vis)
            
    elif args.image_dir:
        # Multiple images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_paths = []
        
        for file in os.listdir(args.image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.image_dir, file))
                
        print(f"Processing {len(image_paths)} images from {args.image_dir}")
        results = inference.predict_batch(image_paths)
        
        # Save results
        output_file = os.path.join(args.output_dir, "batch_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Batch results saved to: {output_file}")
        
        # Print summary
        total_detections = sum(r['num_detections'] for r in results)
        print(f"\nSummary:")
        print(f"Total images processed: {len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections / len(results):.2f}")
        
    else:
        print("Please provide either --image_path or --image_dir")


if __name__ == "__main__":
    main()