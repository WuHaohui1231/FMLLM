import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
import deepspeed
import json
import os

# os.environ["MASTER_PORT"] = "29501"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
# os.environ["TOKENIZERS_PARALLELISM"] = "true" 
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "/model/zhufb/llama32/Llama-3.2-11B-Vision-Instruct"
raw_data_path = "/model/junfeng/GraphRAG-DataSet/news/data/2014/2014-01-01.json"

# Step 1: Load and preprocess the dataset (Assuming tokenized_data is already available)
class FinancialNewsDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Return: Formated data as a list of strings (text)
def preprocess_data(data_path):
    with open(data_path, 'r') as file:
        dataset = json.load(file)

# Preprocess each entry in the dataset
    formatted_data = []
    for entry in dataset:
        # Extract relevant fields and concatenate them into a single string
        formatted_text = f"{entry['Title']}\n{entry['Date']}\n{entry['Text']}"
        formatted_data.append(formatted_text)
    
    return formatted_data


def main():
# Assuming tokenized_data is already loaded and preprocessed
    formatted_data = preprocess_data(raw_data_path)



    # DataLoader for batching
    # train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

    # Step 2: Load the LLaMA Vision 11B model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # print("model max length ", model.model_max_length)
    # model.model_max_length = 2048
    # print("model max length 2 ", model.model_max_length)

    # attention implementation SDPA

    # Move model to GPU(s)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Multi-GPU setup
    print(f"Using {torch.cuda.device_count()} GPUs!")
    # if torch.cuda.device_count() > 1:
        
    #     model = torch.nn.DataParallel(model)

    print("Tokenization")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto') # Max length
    tokenizer.model_max_length = 2048
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    tokenized_data = tokenizer(formatted_data, max_length=2048, padding=True, truncation=True, return_tensors="pt")


    train_dataset = FinancialNewsDataset(tokenized_data)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
        # truncate
    )

    training_args = TrainingArguments(
        output_dir="./llama_3.2_vision_pretraining",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,
        logging_dir='./logs',
        logging_steps=50,
        # fp16=True,
        bf16=True,
        deepspeed='./deepspeed_config.json',  # DeepSpeed config
        #report_to="tensorboard",
    )
    # gradient checkpointing, 

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("TRAINING")

    trainer.train()

# Huggingface SFT

if __name__ == "__main__":
    main()