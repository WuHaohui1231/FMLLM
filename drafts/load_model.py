import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "/model/haohui/LLaMA-Factory/output/fin_lora_pt_1l"
# model_id = "/model/haohui/LLaMA-Factory/output/fin_lora_pt_only_31l"
# model_id = "/model/haohui/LLaMA-Factory-archive/saves/llama3-2_vision/full/llama32-v-11b"
#  model_id = "/model/haohui/LLaMA-Factory/output/fin_lora_pt_31l-30l"
# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # ignore_mismatched_sizes = True
)
# model.tie_weights()
print(model)
# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# adapter_path = "/model/haohui/LLaMA-Factory/saves/llama3-2_vision/lora/pt_fin_31l_lora"

# # Load the fine-tuned LoRA adapter
# model = PeftModel.from_pretrained(model, adapter_path)


# Load image
image_path = "/model/haohui/FMLLM/drafts/kline.png"  # Replace with actual path when needed
image = Image.open(image_path)

# Prepare prompt with image
# Create the conversation input
# question = "What's the highest price of the stock in the image?"
question = "Describe the chart in the image."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    }
]

# Apply the chat template to format the input
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Process the inputs
inputs = processor(image, input_text, return_tensors="pt").to(model.device)


# Generate the answer
output = model.generate(**inputs, max_new_tokens=200)

# Decode and print the answer
answer = processor.decode(output[0], skip_special_tokens=True)
print(answer)


