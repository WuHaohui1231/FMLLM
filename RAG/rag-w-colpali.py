import torch
import base64
import faiss
import os
import json
import uuid
from langchain_core.documents import Document
from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from colpali_embedding import ColPaliEmbeddings
# from langchain_chroma import Chroma

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():

    id_key = "image_id"
    # embedding_function = OpenCLIPEmbeddings()
    embedding_function = ColPaliEmbeddings()

    index = faiss.IndexFlatL2(len(embedding_function.embed_query("hello world")))
    vectorstore = FAISS(
        # collection_name="MMRAG",
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        # distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    # vectorstore = Chroma(
    #     collection_name="mm_rag_clip_photos", embedding_function=embedding_function
    # )

    # questions_path = "/model/haohui/FMLLM/RAG-data/questions-TAT-DQA.json"
    with open("/model/haohui/FMLLM/RAG-data/questions-TAT-DQA.json", "r") as f:
        questions = json.load(f)

    with open("/model/haohui/FMLLM/RAG-data/questions-MMfin.json", "r") as f:
        questions.extend(json.load(f))

    # for question in questions:
    #     image_path = question["image_path"]
    #     image_base64 = encode_image(image_path)

    # image_docs = [
    #     Document(page_content=encode_image(ques["image_path"]), metadata={"doc_id": ques["image_id"], "type": "image"})
    #     for ques in questions
    # ]
    images = [ques["image_path"] for ques in questions]
    ids = [ques["image_id"] for ques in questions]
    metadata = [{"image_id": ques["image_id"]} for ques in questions]

    def encode_image(image_path):
        """Getting the base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    image_base64_list = [encode_image(image_path) for image_path in images]

    images_to_embed = [
        Document(page_content=doc, metadata={id_key: ids[i], "type": "image_to_embed"})
        for i, doc in enumerate(image_base64_list)
    ]

    print("Storing images")
    # vectorstore.add_documents(image_docs)

    # vectorstore.add_images(
    #     uris=images,
    #     ids=ids,
    #     metadatas=metadata,
    #     embedding_function=OpenCLIPEmbeddings(),
    # )

    vectorstore.add_documents(images_to_embed)

    retriever = vectorstore.as_retriever()
    # Get the first document
    # doc = vectorstore.get(include=["documents", "embeddings"])["documents"][0]
    # print(doc)

    correct_count = 0
    total_count = len(questions)

    print("Start evaluation...")
    print(f"Total questions: {total_count}")

    i = 0

    for question_dict in questions:
        question = question_dict["question"]
        image_id = question_dict["image_id"]
        retrieved_content = retriever.invoke(question, k=4)
        for rc in retrieved_content:
            # print("type of rc: ", type(rc))
            # print("metadata: ", rc.metadata)
            retrieved_id = rc.metadata["image_id"]
            print(f"Retrieved ID: {retrieved_id}, Image ID: {image_id}")
            if retrieved_id == image_id:
                print(f"CORRECT!")
                correct_count += 1
                break
            else:
                print(f"WRONG!")

        i += 1
        print(f"Progress: {i}")

    print(f"\nAccuracy: {correct_count / total_count}")


if __name__ == "__main__":
    main()