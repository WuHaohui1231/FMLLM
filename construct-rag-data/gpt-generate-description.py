import json
import os
import base64
import openai
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


prompt = """
You are a financial analyst tasked with describing financial images for retrieval purposes. These descriptions will be embedded and used to retrieve the corresponding raw image.
Instructions:
1. Provide a description of the given image that is well-optimized for retrieval.
2. Identify and mention the names of entities (e.g., companies) referenced in the document image. Avoid using pronouns - always refer to the entity name directly in the description.
3. Include relevant financial terms, metrics, dates, and any visible document types (e.g., earnings report, balance sheet, investor presentation).
4. Aim for specificity and clarity to maximize the match between a future search query and this description. The search query will be a natural-language financial question about the image.
5. The description should have 100 - 300 words.
"""

def generate_description_gpt(image_directory_path, description_file_path):
    # Initialize OpenAI API
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # Prepare the list to store descriptions
    descriptions = []
    len_descriptions = []

    # Iterate over each image in the directory
    for image_filename in os.listdir(image_directory_path):
        #if image_filename.endswith(".png"):
        image_id = os.path.splitext(image_filename)[0]
        image_path = os.path.join(image_directory_path, image_filename)

        base64_image = encode_image(image_path)
        description = image_summarize(base64_image, prompt)

        # Generate description using OpenAI API
        # response = openai.Image.create(
        #     file=open(image_path, "rb"),
        #     model="image-alpha-001"
        # )
        # description = response["data"]["description"]

        print(f"{image_id}: {description}")
        num_words = len(description.split())
        print(num_words)
        len_descriptions.append(num_words)
        # Append the description to the list
        descriptions.append({
            "id": image_id,
            "path": image_path,
            "description": description
        })

        # Sleep for 3 seconds to avoid rate limiting
        time.sleep(3)

    print(f"\nDescription gen done, saving to {description_file_path}")
    # Write the descriptions to a JSON file
    with open(description_file_path, "w") as f:
        json.dump(descriptions, f, indent=4)

    # Save the lengths of descriptions to a JSON file
    len_descriptions_file_path = "MMfin_descriptions_lengths.json"
    with open(len_descriptions_file_path, "w") as f:
        json.dump(len_descriptions, f)
    
    # print(f"Description lengths saved to: {len_descriptions_file_path}")
    # print(f"Average length of descriptions: {sum(len_descriptions) / len(len_descriptions)}")

    return


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
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
    description_file_path = "/model/haohui/FMLLM/RAG-data/descriptions-MMfin.json"
    generate_description_gpt(image_directory_path=image_directory_path, description_file_path=description_file_path)


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