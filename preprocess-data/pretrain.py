import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    default_data_collator
)
import deepspeed
import numpy as np
from typing import Dict, List
import os
import csv
import time
from datetime import datetime

class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        # Create/open the CSV file and write the header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Step', 'Loss', 'Learning Rate', 'Epoch'])
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Only log if loss exists in logs
        if 'loss' not in logs:
            return
            
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract relevant information
        step = state.global_step
        loss = logs.get('loss', 0.0)
        learning_rate = logs.get('learning_rate', 0.0)
        epoch = logs.get('epoch', 0.0)
        
        # Append to the CSV file
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, step, loss, learning_rate, epoch])

def load_and_tokenize_data(json_file: str, tokenizer, max_length: int = 512):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("DATALEN", len(data))

    # Extract text from the JSON objects
    texts = [f"{item['Title']}\n{item['Date']}\n{item['Text']}" for item in data]
    
    # Tokenize all texts
    tokenized_data = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return tokenized_data

class FinancialNewsDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx].clone()
        }

def main():
    # Configuration
    # model_name = "meta-llama/Llama-2-7b"  # Replace with your base model
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # train_file = "2021-01-06.json"
    # train_file = "2022-10-28.json"
    train_file = "partial_data.json"
    # train_file = "partial_financial_news_data.json"
    output_dir = "llm_pretrained_2"
    max_length = 1024

    # os.environ["MASTER_PORT"] = "29501"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'training_loss_{timestamp}.csv'
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation='flash_attention_2',
        # device_map="auto"
    )
    
    print("TOKENIZATION")
    # Load and tokenize data
    tokenized_data = load_and_tokenize_data(train_file, tokenizer, max_length)
    print("END Tokenization")
    # Create dataset
    dataset = FinancialNewsDataset(tokenized_data)

    # print(tokenized_data)
    
    # DeepSpeed configuration
    
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        logging_dir="./logs",
        logging_first_step=True,
        logging_steps=10,
        save_steps=2000,
        save_total_limit=3,
        deepspeed='deepspeed_config.json',
        fp16=False,
        bf16=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        callbacks=[LossLoggingCallback(log_file)]
    )
    
    print("TRAINING")
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()