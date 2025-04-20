import torch
import base64
from io import BytesIO
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from transformers import ColPaliForRetrieval, ColPaliProcessor
from PIL import Image
import numpy as np
import os
from typing import List, Dict
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.docstore.in_memory import InMemoryStore


class ColPaliEmbeddings(Embeddings):
    """Custom embeddings class for ColPali"""
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize with models specialized for different aspects of financial images
        
        Args:
            clip_model_name: HuggingFace model ID for CLIP
            layoutlm_model_name: HuggingFace model ID for LayoutLMv3
            embedding_dim: Dimension of the final embedding
        """

        # 1536 is the dimension of OpenAI text-embedding-3-small
        # self.embedding_dim = embedding_dim
        
        # OpenAI embeddings for text
        # self.text_embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        
        # model_name = "vidore/colqwen2-v1.0"
        model_name = "vidore/colpali-v1.3-hf"

        # ColPali embeddings for image
        self.embedding_model = ColPaliForRetrieval.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        
        
        # Move models to appropriate device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedding_model.to(self.device)


      
    def _process_image(self, image_base64: str) -> Dict[str, torch.Tensor]:
        """Process a single image with both models to extract visual and textual features"""
        # try:
            # Load the image
        image = Image.open(BytesIO(base64.b64decode(image_base64)))


        with torch.no_grad():
            batch_images = self.processor(images=[image], return_tensors="pt").to(self.device)
            image_embeddings = self.embedding_model(**batch_images).embeddings
            # print("type(image_embeddings): ", type(image_embeddings))
            # print("image_embeddings.shape: ", image_embeddings.shape)
            image_embedding = image_embeddings[0].cpu().numpy()
        
        # Combine the embeddings (simple concatenation with normalization)
        # combined = np.concatenate([
        #     clip_embedding / np.linalg.norm(clip_embedding),
        #     image_embedding / np.linalg.norm(image_embedding)
        # ], axis=1)
        
        # Ensure correct output dimension
        # if image_embedding.shape[1] > self.embedding_dim:
        #     # Use PCA-like dimension reduction (simplified version)
        #     image_embedding = image_embedding[:, :self.embedding_dim]
        # elif image_embedding.shape[1] < self.embedding_dim:
        #     # Pad with zeros if needed
        #     padding = np.zeros((image_embedding.shape[0], self.embedding_dim - image_embedding.shape[1]))
        #     image_embedding = np.concatenate([image_embedding, padding], axis=1)
            
        return image_embedding.astype(np.float32)
            
        # except Exception as e:
        #     print(f"Error processing image EMBEDDING: {e}")
        #     # Return zero embedding as fallback
        #     return np.zeros((1, self.embedding_dim)).astype(np.float32)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (image paths)
        
        Args:
            documents: List of image paths to embed
            
        Returns:
            List of embeddings for each document
        """
        embeddings = []
        for doc_str in documents:
            # Check if it's a file path to an image
            # if doc_str[:8] == "isImage;":

                # print("Start image embedding")

            embedding = self._process_image(doc_str)
            embeddings.append(embedding[0].tolist())
                # print("Embedding: ", embedding[0].tolist())

                # print("Image embedding done")

            # else:
            #     # If it's text, use the text embedder
            #     text_embedding = self.text_embedding.embed_documents([doc_str])[0]
            #     # Ensure the embedding has the correct dimension
            #     # if len(text_embedding) > self.embedding_dim:
            #     #     text_embedding = text_embedding[:self.embedding_dim]
            #     # elif len(text_embedding) < self.embedding_dim:
            #     #     padding = [0.0] * (self.embedding_dim - len(text_embedding))
            #     #     text_embedding = text_embedding + padding
            #     embeddings.append(text_embedding)
                
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding
        """
        # Use text embeddings for queries
        with torch.no_grad():
            batch_queries = self.processor(text=[query]).to(self.device)
            text_embedding = self.embedding_model(**batch_queries).embeddings
        
        print("text_embedding.shape: ", text_embedding.shape)
        text_embedding = text_embedding[0][0]
        
        # Ensure the embedding has the correct dimension
        # if len(text_embedding) > self.embedding_dim:
        #     text_embedding = text_embedding[:self.embedding_dim]
        # elif len(text_embedding) < self.embedding_dim:
        #     padding = [0.0] * (self.embedding_dim - len(text_embedding))
        #     text_embedding = text_embedding + padding
            
        return text_embedding.cpu().numpy().tolist()