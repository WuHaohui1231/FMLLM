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
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

descriptions_path = "/model/haohui/FMLLM/rag-data/descriptions.json"
os.environ["OPENAI_API_KEY"] = ""



def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def create_multi_vector_retriever(
    vectorstore, image_descriptions, image_base64s
):
    """
    Create retriever that indexes summaries, but returns raw image_base64s or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_descriptions, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        description_docs = [
            Document(page_content=desc, metadata={id_key: doc_ids[i], "type": "text"})
            for i, desc in enumerate(doc_descriptions)
        ]
        docs = [
            Document(page_content=desc, metadata={id_key: doc_ids[i], "type": "image"})
            for i, desc in enumerate(doc_contents)
        ]
        retriever.vectorstore.add_documents(description_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))

    # Add image_base64s
    if image_descriptions:
        add_documents(retriever, image_descriptions, image_base64s)

    return retriever

def main():
    with open(descriptions_path, 'r') as f:
        descriptions = json.load(f)

    img_descriptions = []
    img_base64_list = []

    for des_img in descriptions:
        image_path = des_img['image_path']
        image_desc = des_img['description']
        image_base64 = encode_image(image_path)
        
        img_base64_list.append(image_base64)
        img_descriptions.append(image_desc)
        # image.show()

    # print("\n\n", img_base64_list)
    # print(img_descriptions)

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    index = faiss.IndexFlatL2(len(embedding_function.embed_query("hello world")))
    # Create a vector store
    vectorstore = FAISS(
        # collection_name="MMRAG",
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        img_descriptions,
        img_base64_list,
    )


    query = "How's the institutional rating of Amazon"
    retrieved_content = retriever_multi_vector_img.invoke(query)
    best_match_image = retrieved_content[0].page_content

    # Import necessary libraries for Llama 3.2 Vision model inference

    
    # Load Llama 3.2 Vision model and tokenizer
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare the prompt combining the query and the image
    prompt = f"Based on the following image, please answer this question: {query}"
    
    # Prepare inputs for the model
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        images=best_match_image,
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nQuery: {query}")
    print(f"\nResponse from Llama 3.2 Vision model:\n{response}")
    
    # print([t.metadata for t in retrieved_content])
    # print("Length: ", len(retrieved_content))
    # first_retrieved = retrieved_content[0]

    # # Save the first retrieved document (base64 encoded image) to a file
    # if retrieved_content and len(retrieved_content) > 0:
    #     # Extract the base64 string from the document
    #     base64_data = retrieved_content[0].page_content
        
    #     # Remove the base64 header if present
    #     if "," in base64_data:
    #         base64_data = base64_data.split(",")[1]
        
    #     # Decode the base64 string to binary data
    #     image_data = base64.b64decode(base64_data)
        
    #     # Save the binary data to a file
    #     output_path = "retrieved_image.png"
    #     with open(output_path, "wb") as f:
    #         f.write(image_data)
        
    #     print(f"First retrieved image saved to {output_path}")
    # else:
    #     print("No images retrieved")



if __name__ == "__main__":
    main()