from io import BytesIO
import torch
import base64
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText

from retrieve import retrieve_best_image, create_multi_vector_retriever, store_data_to_retriever, retrieve_top_k_images
from accelerate import infer_auto_device_map




# Assume these variables are already defined:
# image = PIL.Image object (e.g., Image.open("path/to/image.jpg"))
# question = "What is in this image?"

# Specify the model id for Llama 3.2 Vision


def generate_answer(image, question, model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"):
    # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Load the model and processor from Hugging Face
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Try to resolve the issue "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."

    # if not getattr(model.config, "tie_word_embeddings", False):
    #     model.config.tie_word_embeddings = True

    # # Explicitly tie the weights
    # model.tie_weights()

    # # Now infer the device map (no warning should be triggered)
    # device_map = infer_auto_device_map(model)
    # model.device_map = device_map

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)

    prompt = (
        "You are financial analyst tasking with providing investment advice.\n"
        "You will be given an image usually of charts or graphs.\n"
        "Use this information to answer the user question. \n"
        f"User-provided question: {question}\n\n"
        # "Text and / or tables:\n"
        # f"{formatted_texts}"
    )

    # Create a conversation message with an image and a text question
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Format the conversation into a prompt string expected by the model
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # # Process multiple images and text to create model inputs
    # if isinstance(image, list):
    #     # Handle multiple images
    #     processed_inputs = []
    #     for img in image:
    #         # Process each image individually
    #         processed_input = processor(img, input_text, return_tensors="pt").to(model.device)
    #         processed_inputs.append(processed_input)
        
    #     # Combine the processed inputs
    #     # For Llama 3.2 Vision, we need to concatenate the pixel values and attention masks
    #     # This is a simplified approach - may need adjustment based on model specifics
    #     inputs = processed_inputs[0]
    #     for i in range(1, len(processed_inputs)):
    #         # Concatenate pixel values along the batch dimension
    #         inputs["pixel_values"] = torch.cat([inputs["pixel_values"], processed_inputs[i]["pixel_values"]], dim=0)
    #         # Concatenate attention masks if present
    #         if "attention_mask" in inputs and "attention_mask" in processed_inputs[i]:
    #             inputs["attention_mask"] = torch.cat([inputs["attention_mask"], processed_inputs[i]["attention_mask"]], dim=0)
    # else:
    #     # Process a single image
    
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    # Generate the answer with the model (adjust max_new_tokens as needed)
    output = model.generate(**inputs, max_new_tokens=200)

    # Decode the generated tokens into a string answer
    result = processor.decode(output[0])

    return result

def main():
    query = "What's the current price of Microchip"
    print("Retrieving image...")
    retriever = create_multi_vector_retriever(id_key = "doc_id")
    store_data_to_retriever(retriever)
    retrieved_image_base64 = retrieve_best_image(query, retriever)
    image = Image.open(BytesIO(base64.b64decode(retrieved_image_base64)))
    # retrieved_image_base64 = retrieve_top_k_images(query, retriever, k=2)
    # image = [Image.open(BytesIO(base64.b64decode(img))) for img in retrieved_image_base64]
    print("Retrieved image. Generating answer...")
    answer = generate_answer(image, query)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
