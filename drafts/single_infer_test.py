import torch
from PIL import Image as PIL_Image
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaProcessor

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
    # model_id = "/model/haohui/models/pt-ft/31-30l-nonfreeze-ft"
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the processor (handles both text and image inputs)
    processor = MllamaProcessor.from_pretrained(model_id)
    
    # Load the model with BF16 precision if on GPU to save memory
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    
    # with open(image_path, "rb") as f:
    #     raw_image = PIL_Image.open(f).convert("RGB")
    # Load and preprocess the image
    raw_image = Image.open(image_path).convert("RGB")

    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in two sentences"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,tokenize=False)

    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=512)

    response = processor.decode(output[0])[len(prompt):]

    return response

    # print("text&image_output: ",processor.decode(output[0])[len(prompt):])

    # text_prompt = processor.apply_chat_template(conversation)

    # print(text_prompt)
    
    # Format the prompt with the question
    # prompt = f"<|begin_of_text|><|user|>\n{question}\n<|image|>\n<|end_of_turn|>\n<|assistant|>"
    
    # # Process inputs (handles text and image together)
    # inputs = processor(
    #     images=[image],
    #     text=prompt,
    #     return_tensors="pt"
    # ).to(device)
    
    
    # # Generate the response
    # with torch.no_grad():
    #     output = model.generate(
    #         input_ids=inputs['input_ids'],
    #         attention_mask=inputs['attention_mask'],
    #         max_new_tokens=512,
    #         temperature=0.7,
    #         do_sample=True
    #     )
    
    # # Decode the response and extract only the model's answer
    # response = processor.decode(output[0], skip_special_tokens=False)
    
    # # Extract just the assistant's response
    # assistant_response = response.split("<|assistant|>")[-1].strip()
    # return assistant_response

if __name__ == "__main__":
    # Example usage
    image_path = "/model/haohui/FMLLM/drafts/statistic-7.png"
    question = "What can you see in this image?"
    
    response = run_llama_vision_inference(image_path, question)
    print(f"Question: {question}")
    print(f"Response: {response}")