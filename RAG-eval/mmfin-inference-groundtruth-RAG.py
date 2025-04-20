import json
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
from PIL.Image import Resampling


# prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a financial analyst. You are given a image of a page (or multiple pages concatenated vertically) from a financial report, and a question. Answer the question accurately based on the information shown in the provided image.<|eot_id|><|start_header_id|>user<|end_header_id|>

# <|image|>{}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>"""

system_prompt = "You are a financial analyst. You are given a finance-related image and a question. Answer the question accurately based on the information shown in the image."

def run_llama_vision_inference(model_name_or_path, questions_path, save_path):
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
    model_id = model_name_or_path
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the processor (handles both text and image inputs)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    # Load the model with BF16 precision if on GPU to save memory
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    
    # Load and preprocess the image
    # image = Image.open(image_path).convert("RGB")
    
    # Format the prompt with the question
    # prompt_template = "<|begin_of_text|><|user|>\n{}\n<|image|>\n<|end_of_turn|>\n<|assistant|>"
    # prompt = prompt_template.format(question)

    with open(questions_path, "r") as f:
        questions = json.load(f)
    # Process inputs (handles text and image together)

    QA_pairs = []

    print(f"\nTotal # questions: {len(questions)}\n\n\n")
    i = 1
    for question_dict in questions:
        question = question_dict["question"]
        image_id = question_dict["image_id"]
        image_path = question_dict["image_path"]
        GT_answer = question_dict["GT_answer"]
       

        image = Image.open(image_path).convert("RGB")
        # Resize the image to 1120x1120
        # image = image.resize((1120, 1120), Resampling.LANCZOS)
        # print(image)
    
        # Process inputs (handles text and image together)

        conversation = [
            {
            "role": "system",
            "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
            "role": "user",
            "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,tokenize=False)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        inputs = processor(
            images=[image],
            text=prompt,
            return_tensors="pt"
        ).to(device)
    
        # Generate the response
        with torch.no_grad():
            output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=512)
        
        response = processor.decode(output[0])[(len(prompt)+len("end_header_id|>\n\n")):]
        response = response[:-len("<|eot_id|>")]

    
        # Decode the response and extract only the model's answer
        # response = processor.decode(output[0], skip_special_tokens=False)
    
        # # Extract just the assistant's response
        # assistant_answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        # assistant_answer = assistant_answer.split("<|eot_id|>")[0].strip()
        print(i)
        i += 1
        # print(f"Image ID: {doc_uid}")
        print(f"Question: {question}")
        print(f"Prediction: {response}")
        print("-"*100)

        QA_pairs.append({
            "question": question,
            # "answer": answer,
            # "doc_uid": doc_uid,
            "image_id": image_id,
            "image_path": image_path,
            "GT_answer": GT_answer,
            "prediction": response,
        })

    print(f"Save inference results to {save_path}")
    with open(save_path, "w") as f:
        json.dump(QA_pairs, f, indent=4)
    
    # return assistant_answer

# def generate_answer()

if __name__ == "__main__":
    # Example usage
    model_path = "/model/haohui/models/pt-ft/31-30l-nonfreeze-ft"
    # model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    questions_path = "/model/haohui/FMLLM/RAG-data/QAs-MMfin.json"
    save_path = f"/model/haohui/FMLLM/RAG-eval/mmfin-{model_path[-7:]}-groundtruth-RAG-infer-result.json"
    
    response = run_llama_vision_inference(model_path, questions_path, save_path)
    # print(f"Question: {question}")
    # print(f"Response: {response}")