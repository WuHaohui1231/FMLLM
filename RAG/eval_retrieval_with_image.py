import json
import argparse
from retrieve import create_multi_vector_retriever, store_data_to_retriever, retrieve_best_image, retrieve_top_k_images, retrieve_with_image
# from generate import generate_answer

def main(descriptions_path, questions_path, num_retrieve):

    print(f"Evaluate RAG on {descriptions_path}")

    retriever = create_multi_vector_retriever(id_key = "doc_id")
    
    store_data_to_retriever(retriever, descriptions_path)

    with open(questions_path, "r") as f:
        questions = json.load(f)

    correct_count = 0
    total_count = len(questions)

    print("Start evaluation...")
    print(f"Total questions: {total_count}")

    for _q, image_id in questions.items():
        image_path = "/model/haohui/FMLLM/RAG-data/MMfin-images/" + image_id + ".png"
        retrieved_content = retrieve_with_image(image_path, retriever, num_retrieve)
        for rc in retrieved_content:
            retrieved_id = rc.metadata["doc_id"]
            if retrieved_id == image_id:
                print(f"CORRECT!")
                correct_count += 1
                break
            else:
                print(f"WRONG!")

    print(f"\nAccuracy: {correct_count / total_count}")

if __name__ == "__main__":

    
    def parse_args():
        parser = argparse.ArgumentParser(description="Evaluate retrieval performance on different datasets")
        parser.add_argument("--dataset", type=str, default="mmfin", 
                           help="Dataset name to evaluate (default: MMfin)")
        parser.add_argument("--num_retrieve", type=int, default=1, 
                           help="Number of retrieve images (default: 1)")
        return parser.parse_args()
    
    args = parse_args()
    
    dataset_to_descriptions_path = {
        "mmfin": "/model/haohui/FMLLM/RAG-data/descriptions-MMfin.json",
        "tatdqa": "/model/haohui/FMLLM/RAG-data/descriptions-TAT-DQA.json",
    }
    dataset_to_questions_path = {
        "mmfin": "/model/haohui/FMLLM/RAG-data/questions-MMfin.json",
        "tatdqa": "/model/haohui/FMLLM/RAG-data/questions-TAT-DQA.json",
    }

    descriptions_path = dataset_to_descriptions_path[args.dataset]
    questions_path = dataset_to_questions_path[args.dataset]
    num_retrieve = args.num_retrieve
    
    
    main(descriptions_path, questions_path, num_retrieve)