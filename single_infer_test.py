import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

def run_llama_vision_inference(image_path, question):
    """
    Run inference on Llama 3.2 Vision model with an image and question.
    
    Args:
        image_path (str): Path to the image file
        question (str): Question about the image
        
    Returns:
        str: Model's response
    """
    # Load the model and processor
    model_id = "/model/haohui/LLaMA-Factory/saves/llama3-2_vision/full/llama32-v-11b"
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the processor (handles both text and image inputs)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load the model with BF16 precision if on GPU to save memory
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    
    # Format the prompt with the question
    prompt = f"<|begin_of_text|><|user|>\n{question}\n<|image|>\n<|end_of_turn|>\n<|assistant|>"
    
    # Process inputs (handles text and image together)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)
    
    # Generate the response
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode the response and extract only the model's answer
    response = processor.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|assistant|>")[-1].strip()
    return assistant_response

if __name__ == "__main__":
    # Example usage
    image_path = "/model/haohui/FMLLM/test_img.jpg"
    question = "What can you see in this image?"
    
    response = run_llama_vision_inference(image_path, question)
    print(f"Question: {question}")
    print(f"Response: {response}")