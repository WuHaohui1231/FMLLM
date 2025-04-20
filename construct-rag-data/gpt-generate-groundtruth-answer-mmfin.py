import json
import os
import base64
import openai
from openai import OpenAI
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


system_prompt = """
You are a financial analyst. 
I will give you a financial question and a finance-related image (e.g. a chart, a table, a part from financial report, etc.).
Answer the question based on the content of the image. Please give a concise answer."""

def generate_answer_gpt(question_file_path, save_path):
    # Initialize OpenAI API
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # Prepare the list to store descriptions
    with open(question_file_path, "r") as f:
        questions = json.load(f)


    QAs = []

    client = OpenAI()
    
    for question_dict in questions:
        question_id = question_dict['question_id']
        question = question_dict['question']
        image_id = question_dict['image_id']
        image_path = question_dict['image_path']

        img_base64 = encode_image(image_path)

        messages = [
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": system_prompt},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_base64}"
                    },
                ],
            }
        ]

        # prompt = prompt_template.format(question=question, image_path=image_path)
        try:
            response = client.responses.create(
                model="gpt-4o",  # hypothetical; use the actual model name if/when available
                input=messages,
                max_output_tokens=512
            )
            # Print the assistant's answer
            GT_answer = response.output[0].content[0].text
            # print(GT_answer)
        except Exception as e:
            print("An error occurred:", e)

        # GT_answer = answer_VQA_gpt(img_base64, question)

        print(f"image_id: {image_id}\nquestion: {question}\nGT_answer: {GT_answer}")

        QAs.append({
            "question": question,
            "question_id": question_id,
            "image_id": image_id,
            "image_path": image_path,
            "GT_answer": GT_answer,
        })
        # Sleep for 3 seconds to avoid rate limiting
        # time.sleep(3)

    # print(f"\nQuestion gen done, saving to {question_file_path}")
    # Write the descriptions to a JSON file
    with open(save_path, "w") as f:
        json.dump(QAs, f, indent=4)

    # Save the lengths of descriptions to a JSON file
    # len_questions_file_path = question_file_path.replace(".json", "_lengths.json")
    # with open(len_questions_file_path, "w") as f:
    #     json.dump(len_questions, f)
    
    # print(f"Description lengths saved to: {len_descriptions_file_path}")
    # print(f"Average length of descriptions: {sum(len_descriptions) / len(len_descriptions)}")

    return


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# def answer_VQA_gpt(img_base64, prompt):
#     """Make image summary"""
#     chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)

#     msg = chat.invoke(
#         [
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{img_base64}",
#                             "detail": "high"
#                         },
#                     },
#                 ]
#             )
#         ]
#     )
#     return msg.content

if __name__ == "__main__":
    # image_directory_path = "/model/haohui/FMLLM/RAG-data/MMfin-images"
    question_file_path   = "/model/haohui/FMLLM/RAG-data/questions-MMfin.json"
    save_path = "/model/haohui/FMLLM/RAG-data/QAs-MMfin.json"
    generate_answer_gpt(question_file_path=question_file_path, save_path=save_path)


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