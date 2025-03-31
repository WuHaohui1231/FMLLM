import torch
import base64
from io import BytesIO
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from transformers import AutoProcessor, LayoutLMv3Model
from PIL import Image
import numpy as np
import os
from typing import List, Dict

# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.docstore.in_memory import InMemoryStore


class FinancialMultimodalEmbeddings(Embeddings):
    """Custom embeddings class that combines CLIP for visual understanding and LayoutLMv3 for text-in-image understanding"""
    
    def __init__(self, 
                 layoutlm_model_name: str = "microsoft/layoutlmv3-base",
                 embedding_dim: int = 1536):
        """
        Initialize with models specialized for different aspects of financial images
        
        Args:
            clip_model_name: HuggingFace model ID for CLIP
            layoutlm_model_name: HuggingFace model ID for LayoutLMv3
            embedding_dim: Dimension of the final embedding
        """

        # 1536 is the dimension of OpenAI text-embedding-3-small
        self.embedding_dim = embedding_dim
        
        # Load CLIP for visual features
        self.openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Load LayoutLMv3 for text-in-image understanding
        self.layoutlm_model = LayoutLMv3Model.from_pretrained(layoutlm_model_name)
        self.layoutlm_processor = AutoProcessor.from_pretrained(layoutlm_model_name)
        
        
        # Move models to appropriate device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layoutlm_model.to(self.device)
        
    def _process_image(self, image_base64: str) -> Dict[str, torch.Tensor]:
        """Process a single image with both models to extract visual and textual features"""
        # try:
            # Load the image
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
        
        print("Start embedding with LayoutLMv3")
        # Get CLIP embeddings for visual features
        layoutlm_inputs = self.layoutlm_processor(
            text=None,
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            layoutlm_outputs = self.layoutlm_model(**layoutlm_inputs)
            # Extract the image embeddings from the last hidden state
            # Taking the first token's embedding as the image representation
            layoutlm_embedding = layoutlm_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        print("LayoutLMv3 embedding done")

        
        # Combine the embeddings (simple concatenation with normalization)
        # combined = np.concatenate([
        #     clip_embedding / np.linalg.norm(clip_embedding),
        #     layoutlm_embedding / np.linalg.norm(layoutlm_embedding)
        # ], axis=1)
        
        # Ensure correct output dimension
        if layoutlm_embedding.shape[1] > self.embedding_dim:
            # Use PCA-like dimension reduction (simplified version)
            layoutlm_embedding = layoutlm_embedding[:, :self.embedding_dim]
        elif layoutlm_embedding.shape[1] < self.embedding_dim:
            # Pad with zeros if needed
            padding = np.zeros((layoutlm_embedding.shape[0], self.embedding_dim - layoutlm_embedding.shape[1]))
            layoutlm_embedding = np.concatenate([layoutlm_embedding, padding], axis=1)
            
        return layoutlm_embedding.astype(np.float32)
            
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
            if doc_str[:8] == "isImage;":
                embedding = self._process_image(doc_str[8:])
                embeddings.append(embedding[0].tolist())
            else:
                # If it's text, use the text embedder
                text_embedding = self.openai_embeddings.embed_documents([doc_str])[0]
                # Ensure the embedding has the correct dimension
                # if len(text_embedding) > self.embedding_dim:
                #     text_embedding = text_embedding[:self.embedding_dim]
                # elif len(text_embedding) < self.embedding_dim:
                #     padding = [0.0] * (self.embedding_dim - len(text_embedding))
                #     text_embedding = text_embedding + padding
                embeddings.append(text_embedding)
                
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
        text_embedding = self.openai_embeddings.embed_query(query)
        
        # Ensure the embedding has the correct dimension
        # if len(text_embedding) > self.embedding_dim:
        #     text_embedding = text_embedding[:self.embedding_dim]
        # elif len(text_embedding) < self.embedding_dim:
        #     padding = [0.0] * (self.embedding_dim - len(text_embedding))
        #     text_embedding = text_embedding + padding
            
        return text_embedding