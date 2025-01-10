import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    default_data_collator
)
import deepspeed
import numpy as np
from typing import Dict, List
import statistics

def load_and_tokenize_data(json_file: str, tokenizer, max_length: int = 512):
    # Load the JSON data
    print("start load")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("DATALEN", len(data))

    # Extract text from the JSON objects
    texts = [f"{item['Title']}\n{item['Date']}\n{item['Text']}" for item in data]

    lengths = []

    for i, piece in enumerate(texts):
        temp = tokenizer(piece)
        lengths.append(len(temp['input_ids']))
        if i % 1000 == 0:
            avg_len = statistics.fmean(lengths)
            median = statistics.median(lengths)
            print("AVG length ", avg_len)
            print("Median ", median)

    print("Final AVG length ", statistics.fmean(lengths))
    print("Final Median ", statistics.median(lengths))


    # Tokenize all texts
    # tokenized_data = tokenizer(
    #     texts,
    #     max_length=max_length,
    #     # padding='max_length',
    #     truncation=True,
    #     return_tensors='pt'
    # )
    
    # return tokenized_data

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
    train_file = "consolidated_financial_news_data.json"
    output_dir = "llm_pretrained"
    max_length = 1024
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     # device_map="auto"
    # )
    
    print("TOKENIZATION")
    # Load and tokenize data
    tokenized_data = load_and_tokenize_data(train_file, tokenizer, max_length)
    print("END Tokenization")
    # Create dataset
    # dataset = FinancialNewsDataset(tokenized_data)

    print(tokenized_data['input_ids'].shape)
    
    # DeepSpeed configuration
    
    
    # Training arguments


if __name__ == "__main__":
    main()