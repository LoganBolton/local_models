import os
import argparse
import torch
import json
import wandb
from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from data_loader import create_dataloaders
import bitsandbytes as bnb


class LicensePlateTrainer:
    """Trainer for fine-tuning Qwen2.5-VL with LoRA on license plate detection."""
    
    def __init__(self, config: dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.setup_logging()
        self.setup_model_and_processor()
        self.setup_lora()
        self.setup_data()
        
    def setup_logging(self):
        """Setup logging and experiment tracking."""
        if self.config['use_wandb']:
            wandb.init(
                project=self.config['wandb_project'],
                name=f"qwen2.5-vl-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
            
    def setup_model_and_processor(self):
        """Load model and processor."""
        print(f"Loading model: {self.config['model_name']}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True
        )
        
        # Load model with quantization if specified
        if self.config['use_4bit']:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        # Set pad token if not set
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
    def setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        print("Setting up LoRA configuration...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=self.config['lora_target_modules'],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        
    def setup_data(self):
        """Setup datasets and data loaders."""
        print("Loading datasets...")
        
        self.train_dataset, self.val_dataset, self.test_dataset = create_dataloaders(
            train_annotations=self.config['train_annotations'],
            val_annotations=self.config['val_annotations'],
            test_annotations=self.config['test_annotations'],
            images_dir=self.config['images_dir'],
            processor=self.processor,
            batch_size=self.config['batch_size'],
            max_length=self.config['max_length']
        )
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        
    def train(self):
        """Train the model."""
        print("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            eval_steps=self.config['eval_steps'],
            eval_strategy="steps",
            save_total_limit=self.config['save_total_limit'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config['use_wandb'] else None,
            run_name=f"qwen2.5-vl-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=True,
            gradient_checkpointing=True,
            warmup_ratio=self.config['warmup_ratio'],
            lr_scheduler_type="cosine",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.processor.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(self.config['output_dir'])
        
        print(f"Training completed! Model saved to: {self.config['output_dir']}")
        
    def evaluate(self):
        """Evaluate the model on test set."""
        print("Evaluating model...")
        
        # Load best model for evaluation
        model_path = os.path.join(self.config['output_dir'], "pytorch_model.bin")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
        
        # Simple evaluation on a few samples
        test_samples = 5
        correct = 0
        
        with torch.no_grad():
            for i in range(min(test_samples, len(self.test_dataset))):
                sample = self.test_dataset[i]
                
                # Get model prediction
                outputs = self.model.generate(
                    input_ids=sample['input_ids'].unsqueeze(0),
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.1,
                )
                
                # Decode prediction
                prediction = self.processor.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                print(f"Sample {i+1}:")
                print(f"Prediction: {prediction}")
                print("-" * 50)
                
        print("Evaluation completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()
    
    # Default configuration
    default_config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "train_annotations": "dataset/_annotations.train.jsonl",
        "val_annotations": "dataset/_annotations.valid.jsonl", 
        "test_annotations": "dataset/_annotations.test.jsonl",
        "images_dir": "dataset",
        "output_dir": "./results",
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_length": 512,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 3,
        "warmup_ratio": 0.1,
        "use_4bit": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "use_wandb": False,
        "wandb_project": "qwen2.5-vl-lora",
        "seed": 42,
    }
    
    # Load config if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Update default config with loaded config
        default_config.update(config)
    else:
        config = default_config
        # Save default config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {args.config}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = LicensePlateTrainer(config)
    
    # Train
    trainer.train()
    
    # Evaluate
    trainer.evaluate()


if __name__ == "__main__":
    main()