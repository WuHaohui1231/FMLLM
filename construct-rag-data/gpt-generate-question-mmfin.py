import json
import os
import base64
import openai
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


prompt = """
You are a financial analyst assistant. 
I will give you a finance-related image such as financial charts (e.g. stock price candlestick chart, bar chart, etc.), tables (e.g. financial statements, etc.), textual reports, or a combination of them.\
Generate one question based on the financial information shown in the image.
Instructions:
1. Avoid using vague terms that are not specific to a particular image, such as "in the image", "this image", "according to the provided image", "shown on the ... chart", etc. \
You should assume that the user does not have the image when you ask the question. They will retrieve the image based on your question (and answer your question base on the retrieved image).
2. For clarity and optimization for retrieval, the question should include key information in the image, such as the name of the entities (e.g. companies, etc.), the time period, the financial metrics and terminologies, etc.
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
        question_id = image_id + '-q1'
        print(f"{image_id}: {question}")
        num_words = len(question.split())
        print(num_words)
        len_questions.append(num_words)

        questions.append({
            "question": question,
            "image_id": image_id,
            "image_path": image_path,
            "question_id": question_id
        })

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


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def question_image(img_base64, prompt):
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
    image_directory_path = "/model/haohui/FMLLM/RAG-data/MMfin-images"
    question_file_path   = "/model/haohui/FMLLM/RAG-data/questions-MMfin.json"
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