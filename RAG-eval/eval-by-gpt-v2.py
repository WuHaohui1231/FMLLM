import json
import os
import base64
import openai
from openai import OpenAI
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


system_prompt = """You are a grader for a financial visual question answering task. 
I will give you a financial question and a finance-related image (e.g. a chart, a table, a part from financial report, etc.), as well as the predicted answer by a model.
Please grade if the predicted answer is correct. If correct, output 1, otherwise output 0.
You output should contain only one number (1 or 0) indicating if the predicted answer is correct."""


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def evaluate_result_gpt(result_path):
    # Initialize OpenAI API
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # Prepare the list to store descriptions
    with open(result_path, "r") as f:
        QAs = json.load(f)

    client = OpenAI()
    num_QAs = len(QAs)
    correct_count = 0
    
    for question_dict in QAs:
        # question_id = question_dict['question_id']
        question = question_dict['question']
        image_id = question_dict['image_id']
        image_path = question_dict['image_path']
        
        if "prediction" in question_dict:
            prediction = question_dict['prediction']
        elif "answer" in question_dict:
            prediction = question_dict['answer']

        img_base64 = encode_image(image_path)

        QA = f"Question: {question}\nModel's Predicted Answer: {prediction}\n"

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
                    {"type": "input_text", "text": QA},
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
            grade = response.output[0].content[0].text
            # if "1" in grade:
            #     correct_count += 1
            
            # Extract the grade (1 or 0) from the response
            if "grade: 1" in grade.lower() or grade.strip() == "1":
                # grade_value = 1
                correct_count += 1
            elif "grade: 0" in grade.lower() or grade.strip() == "0":
                # grade_value = 0
                pass
            else:
                # Try to find any number in the response
                import re
                numbers = re.findall(r'\d+', grade)
                if numbers and int(numbers[0]) == 1:
                    # grade_value = 1
                    correct_count += 1
                else:
                    # grade_value = 0
                    pass
            # print(GT_answer)
        except Exception as e:
            print("An error occurred:", e)

        # GT_answer = answer_VQA_gpt(img_base64, question)

        print(f"image_id: {image_id}\nquestion: {question}\ngrade: {grade}")
        if "1" in grade:
            print("correct")


    final_grade = correct_count / num_QAs
    print("\n\n")
    print(result_path)
    print(f"\nFinal grade: {final_grade}\n")

    #     QAs.append({
    #         "question": question,
    #         "question_id": question_id,
    #         "image_id": image_id,
    #         "image_path": image_path,
    #         "GT_answer": GT_answer,
    #     })
    #     # Sleep for 3 seconds to avoid rate limiting
    #     # time.sleep(3)

    # # print(f"\nQuestion gen done, saving to {question_file_path}")
    # # Write the descriptions to a JSON file
    # with open(save_path, "w") as f:
    #     json.dump(QAs, f, indent=4)

    # Save the lengths of descriptions to a JSON file
    # len_questions_file_path = question_file_path.replace(".json", "_lengths.json")
    # with open(len_questions_file_path, "w") as f:
    #     json.dump(len_questions, f)
    
    # print(f"Description lengths saved to: {len_descriptions_file_path}")
    # print(f"Average length of descriptions: {sum(len_descriptions) / len(len_descriptions)}")

    return







if __name__ == "__main__":
    # image_directory_path = "/model/haohui/FMLLM/RAG-data/MMfin-images"
    # question_file_path   = "/model/haohui/FMLLM/RAG-data/questions-MMfin.json"
    # save_path = "/model/haohui/FMLLM/RAG-data/QAs-MMfin.json"
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate answers using GPT')

    parser.add_argument('--result_path', type=str,
                        help='Path to save evaluation results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update paths based on arguments
    # question_file_path = args.question_file
    # save_path = args.save_path
    result_path = args.result_path
    evaluate_result_gpt(result_path=result_path)


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