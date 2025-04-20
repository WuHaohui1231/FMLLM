import json
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
from PIL.Image import Resampling
import base64
from io import BytesIO

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

import sys
sys.path.insert(0, '/model/haohui/FMLLM/RAG')
from retrieve_text_only import create_multi_vector_retriever, retrieve_best_image, store_data_to_retriever


# prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a financial analyst. You are given a image of a page (or multiple pages concatenated vertically) from a financial report, and a question. Answer the question accurately based on the information shown in the provided image.<|eot_id|><|start_header_id|>user<|end_header_id|>

# <|image|>{}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>"""

system_prompt = "You are a financial analyst. You are given a finance-related image and a question. Answer the question accurately based on the information shown in the image."

def inference_textInput_with_RAG(model_name_or_path, questions_path, descriptions_path, save_path):
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


    retriever = create_multi_vector_retriever()
    store_data_to_retriever(retriever, descriptions_path)


    QA_pairs_w_predictions = []

    print(f"\nTotal # questions: {len(questions)}\n\n\n")
    i = 1
    for question_dict in questions:
        question = question_dict["question"]

        # Not actually used for inference. Jut put in output file for easy evaluation later
        image_id = question_dict["image_id"]
        image_path = question_dict["image_path"]
        GT_answer = question_dict["GT_answer"]

        retrieved_image_document = retrieve_best_image(question, retriever)
        retrieved_image_base64 = retrieved_image_document.page_content
        retrieved_image_id = retrieved_image_document.metadata["doc_id"]

        retrieved_image = Image.open(BytesIO(base64.b64decode(retrieved_image_base64)))


    
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

        # inputs = processor(images=retrieved_image, text=prompt, return_tensors="pt").to(model.device)
        inputs = processor(
            images=retrieved_image,
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
        print(f"Image ID: {image_id}. Retrieved ID: {retrieved_image_id}")
        print(f"Prediction: {response}")
        print("-"*100)

        QA_pairs_w_predictions.append({
            "question": question,
            # "doc_uid": doc_uid,
            "image_id": image_id,
            "retrieved_image_id": retrieved_image_id,
            "image_path": image_path,
            "GT_answer": GT_answer,
            "prediction": response,
        })

    print(f"Save inference results to {save_path}")
    with open(save_path, "w") as f:
        json.dump(QA_pairs_w_predictions, f, indent=4)
    
    # return assistant_answer

# def generate_answer()

if __name__ == "__main__":
    # Example usage
    # model_path = "/model/haohui/models/pt-ft/31-30l-nonfreeze-ft"
    # model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # questions_path = "/model/haohui/FMLLM/RAG-data/questions-TAT-DQA.json"
    # save_path = f"/model/haohui/FMLLM/RAG-eval/tatdqa-groundtruth-RAG-infer-result.json"

    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run inference with RAG')
    parser.add_argument('--dataset', type=str,
                        help='Path to the dataset containing questions')
    parser.add_argument('--model', type=str,
                        help='Model name or path')
    
    
    # Parse arguments
    args = parser.parse_args()

    dataset_to_descriptions_path = {
        "mmfin": "/model/haohui/FMLLM/RAG-data/descriptions-MMfin.json",
        "tatdqa": "/model/haohui/FMLLM/RAG-data/descriptions-TAT-DQA.json",
    }
    dataset_to_questions_path = {
        "mmfin": "/model/haohui/FMLLM/RAG-data/QAs-MMfin.json",
        "tatdqa": "/model/haohui/FMLLM/RAG-data/QAs-TAT-DQA.json",
    }

    descriptions_path = dataset_to_descriptions_path[args.dataset]
    questions_path = dataset_to_questions_path[args.dataset]
    model_path = args.model
    save_path = f"/model/haohui/FMLLM/RAG-eval/{args.dataset}-{model_path[-7:]}-RAG-infer-result.json"
    # Update questions_path if dataset argument is provided
    # questions_path = args.dataset
    
    response = inference_textInput_with_RAG(model_path, questions_path, descriptions_path, save_path)
    # print(f"Question: {question}")
    # print(f"Response: {response}")