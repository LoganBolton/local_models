import json
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class LicensePlateDataset(Dataset):
    """Dataset for loading license plate detection data in PaliGemma format."""
    
    def __init__(self, annotations_file: str, images_dir: str, processor, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            annotations_file: Path to the JSONL annotations file
            images_dir: Directory containing the images
            processor: Vision-language model processor
            max_length: Maximum sequence length for text
        """
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length
        
        # Load annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSONL file."""
        annotations = []
        with open(self.annotations_file, 'r') as f:
            for line in f:
                annotations.append(json.loads(line.strip()))
        return annotations
    
    def _parse_bounding_boxes(self, suffix: str) -> List[Tuple[int, int, int, int]]:
        """
        Parse bounding boxes from PaliGemma format suffix.
        
        Args:
            suffix: String like "<loc0505><loc0381><loc0717><loc0615> License-Plate"
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        if not suffix.strip():
            return []
            
        boxes = []
        # Split by " ; " for multiple objects
        parts = suffix.split(' ; ')
        
        for part in parts:
            # Find all location tokens
            loc_pattern = r'<loc(\d{4})>'
            matches = re.findall(loc_pattern, part)
            
            if len(matches) >= 4:
                # Convert from 0-1000 scale to actual coordinates
                x1, y1, x2, y2 = [int(m) for m in matches[:4]]
                boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def _format_answer(self, suffix: str) -> str:
        """
        Format the answer in a consistent way for training.
        
        Args:
            suffix: Original suffix from annotations
            
        Returns:
            Formatted answer string
        """
        if not suffix.strip():
            return "No license plates detected."
        
        # Parse bounding boxes
        boxes = self._parse_bounding_boxes(suffix)
        
        if not boxes:
            return "No license plates detected."
        
        # Format as natural language with coordinates
        if len(boxes) == 1:
            x1, y1, x2, y2 = boxes[0]
            return f"<loc{x1:04d}><loc{y1:04d}><loc{x2:04d}><loc{y2:04d}> License-Plate"
        else:
            formatted_boxes = []
            for x1, y1, x2, y2 in boxes:
                formatted_boxes.append(f"<loc{x1:04d}><loc{y1:04d}><loc{x2:04d}><loc{y2:04d}> License-Plate")
            return " ; ".join(formatted_boxes)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with processed inputs
        """
        annotation = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, annotation['image'])
        image = Image.open(image_path).convert('RGB')
        
        # Create conversation format
        question = annotation['prefix']  # "detect License-Plate"
        answer = self._format_answer(annotation['suffix'])
        
        # Format as conversation
        conversation = f"Question: {question}\nAnswer: {answer}"
        
        # Process with vision-language processor
        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Remove batch dimension since DataLoader will add it
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs


def create_dataloaders(
    train_annotations: str,
    val_annotations: str,
    test_annotations: str,
    images_dir: str,
    processor,
    batch_size: int = 4,
    max_length: int = 512
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test datasets.
    
    Args:
        train_annotations: Path to training annotations
        val_annotations: Path to validation annotations  
        test_annotations: Path to test annotations
        images_dir: Directory containing images
        processor: Vision-language model processor
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = LicensePlateDataset(
        train_annotations, images_dir, processor, max_length
    )
    
    val_dataset = LicensePlateDataset(
        val_annotations, images_dir, processor, max_length
    )
    
    test_dataset = LicensePlateDataset(
        test_annotations, images_dir, processor, max_length
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Create dataset
    dataset = LicensePlateDataset(
        annotations_file="dataset/_annotations.train.jsonl",
        images_dir="dataset",
        processor=processor
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")