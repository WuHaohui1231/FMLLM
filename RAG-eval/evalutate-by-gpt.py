import json
import os
import base64
import openai
import time
from PIL import Image
from PIL.Image import Resampling
from io import BytesIO

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


prompt = """
You are a financial expert who is well-versed in various financial charts and has extensive financial knowledge. You are now evaluting the financial visual question answering performance of an AI model.
Now you are given a finance-related picture (e.g. a chart, a table, a page from financial report, etc.), corresponding question, ground truth answer, and the model's prediction.
Compare the ground truth and prediction from AI models.
The final answer must be an integer score. If a decimal is obtained, it can be rounded.
You output should only contain a single number, which is the final correctness score.
 
"""

def generate_question_gpt(image_directory_path, question_file_path):
    # Initialize OpenAI API
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # Prepare the list to store descriptions
    questions = []
    len_questions = []

    # Iterate over each image in the directory
    for image_filename in os.listdir(image_directory_path):
        #if image_filename.endswith(".png"):
        image_id = os.path.splitext(image_filename)[0]
        image_path = os.path.join(image_directory_path, image_filename)

        base64_image = encode_image(image_path)
        question = question_image(base64_image, prompt)

        # Generate description using OpenAI API
        # response = openai.Image.create(
        #     file=open(image_path, "rb"),
        #     model="image-alpha-001"
        # )
        # description = response["data"]["description"]

        print(f"{image_id}: {question}")
        num_words = len(question.split())
        print(num_words)
        len_questions.append(num_words)
        
        questions.append({
            "question": question,
            "image_id": image_id,
            "image_path": image_path,
            "question_id": image_id + "-q1"
        })
        # Append the description to the list
        # questions.append({
        #     "id": image_id,
        #     "path": image_path,
        #     "question": question
        # })

        # Sleep for 3 seconds to avoid rate limiting
        # time.sleep(3)

    print(f"\nQuestion gen done, saving to {question_file_path}")
    # Write the descriptions to a JSON file
    with open(question_file_path, "w") as f:
        json.dump(questions, f, indent=4)

    # Save the lengths of descriptions to a JSON file
    len_questions_file_path = question_file_path.replace(".json", "_lengths.json")
    with open(len_questions_file_path, "w") as f:
        json.dump(len_questions, f)
    
    # print(f"Description lengths saved to: {len_descriptions_file_path}")
    # print(f"Average length of descriptions: {sum(len_descriptions) / len(len_descriptions)}")

    return


def encode_image(image_path, max_size=(2048, 2048)):
    """Resize and encode the image to base64."""
    with Image.open(image_path) as img:
        # Resize the image to fit within the max_size while maintaining aspect ratio
        img.thumbnail(max_size, Resampling.LANCZOS)
        
        # Save the resized image to a bytes buffer
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode the image to base64
        return base64.b64encode(buffer.read()).decode("utf-8")


def evaluate_prediction(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                            "detail": "high"
                        },
                    },
                ]
            )
        ]
    )
    return msg.content

if __name__ == "__main__":
    image_directory_path = "/model/haohui/FMLLM/RAG-data/TAT-DQA-images"
    question_file_path   = "/model/haohui/FMLLM/RAG-data/questions-TAT-DQA.json"

    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate answers')
    parser.add_argument('--result_path', type=str,
                        help='Path to save the generated questions')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update question_file_path if result_path is provided
    question_file_path = args.result_path
    generate_question_gpt(image_directory_path=image_directory_path, question_file_path=question_file_path)


# def generate_img_summaries(path):
#     """
#     Generate summaries and base64 encoded strings for images
#     path: Path to list of .jpg files extracted by Unstructured
#     """

#     # Store base64 encoded images
#     img_base64_list = []

#     # Store image summaries
#     image_summaries = []

#     # Prompt
#     prompt = """You are an assistant tasked with summarizing images for retrieval. \
#     These summaries will be embedded and used to retrieve the raw image. \
#     Give a concise summary of the image that is well optimized for retrieval."""

#     # Apply to images
#     for img_file in sorted(os.listdir(path)):
#         if img_file.endswith(".jpg"):
#             img_path = os.path.join(path, img_file)
#             base64_image = encode_image(img_path)
#             img_base64_list.append(base64_image)
#             image_summaries.append(image_summarize(base64_image, prompt))

#     return img_base64_list, image_summaries