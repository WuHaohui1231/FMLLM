import os
import glob
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from transformers import CLIPProcessor, CLIPModel, LayoutLMv3Processor, LayoutLMv3Model, AutoProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings

class FinancialMultimodalEmbeddings(Embeddings):
    """Custom embeddings class that combines CLIP for visual understanding and LayoutLMv3 for text-in-image understanding"""
    
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", 
                 layoutlm_model_name: str = "microsoft/layoutlmv3-base",
                 embedding_dim: int = 768):
        """
        Initialize with models specialized for different aspects of financial images
        
        Args:
            clip_model_name: HuggingFace model ID for CLIP
            layoutlm_model_name: HuggingFace model ID for LayoutLMv3
            embedding_dim: Dimension of the final embedding
        """
        self.embedding_dim = embedding_dim
        
        # Load CLIP for visual features
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Load LayoutLMv3 for text-in-image understanding
        self.layoutlm_model = LayoutLMv3Model.from_pretrained(layoutlm_model_name)
        self.layoutlm_processor = AutoProcessor.from_pretrained(layoutlm_model_name)
        
        # Text embeddings for queries
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Move models to appropriate device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.layoutlm_model.to(self.device)
        
    def _process_image(self, image_path: str) -> Dict[str, torch.Tensor]:
        """Process a single image with both models to extract visual and textual features"""
        try:
            # Load the image
            image = Image.open(image_path).convert("RGB")
            
            # Get CLIP embeddings for visual features
            clip_inputs = self.clip_processor(
                text=None,
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                clip_outputs = self.clip_model.get_image_features(**clip_inputs)
                clip_embedding = clip_outputs.cpu().numpy()
            
            # Get LayoutLMv3 embeddings for document understanding (text + layout)
            layoutlm_inputs = self.layoutlm_processor(
                image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                layoutlm_outputs = self.layoutlm_model(**layoutlm_inputs)
                layoutlm_embedding = layoutlm_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Combine the embeddings (simple concatenation with normalization)
            combined = np.concatenate([
                clip_embedding / np.linalg.norm(clip_embedding),
                layoutlm_embedding / np.linalg.norm(layoutlm_embedding)
            ], axis=1)
            
            # Ensure correct output dimension
            if combined.shape[1] > self.embedding_dim:
                # Use PCA-like dimension reduction (simplified version)
                combined = combined[:, :self.embedding_dim]
            elif combined.shape[1] < self.embedding_dim:
                # Pad with zeros if needed
                padding = np.zeros((combined.shape[0], self.embedding_dim - combined.shape[1]))
                combined = np.concatenate([combined, padding], axis=1)
                
            return combined.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return zero embedding as fallback
            return np.zeros((1, self.embedding_dim)).astype(np.float32)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (image paths)
        
        Args:
            documents: List of image paths to embed
            
        Returns:
            List of embeddings for each document
        """
        embeddings = []
        for doc in documents:
            # Check if it's a file path to an image
            if os.path.isfile(doc) and doc.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                embedding = self._process_image(doc)
                embeddings.append(embedding[0].tolist())
            else:
                # If it's text, use the text embedder
                text_embedding = self.text_embeddings.embed_documents([doc])[0]
                # Ensure the embedding has the correct dimension
                if len(text_embedding) > self.embedding_dim:
                    text_embedding = text_embedding[:self.embedding_dim]
                elif len(text_embedding) < self.embedding_dim:
                    padding = [0.0] * (self.embedding_dim - len(text_embedding))
                    text_embedding = text_embedding + padding
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
        text_embedding = self.text_embeddings.embed_query(query)
        
        # Ensure the embedding has the correct dimension
        if len(text_embedding) > self.embedding_dim:
            text_embedding = text_embedding[:self.embedding_dim]
        elif len(text_embedding) < self.embedding_dim:
            padding = [0.0] * (self.embedding_dim - len(text_embedding))
            text_embedding = text_embedding + padding
            
        return text_embedding


class FinancialImageIndexer:
    """Indexes financial images for retrieval"""
    
    def __init__(self, embedding_model: Embeddings = None):
        """
        Initialize the indexer
        
        Args:
            embedding_model: Custom embedding model to use
        """
        if embedding_model is None:
            self.embedding_model = FinancialMultimodalEmbeddings()
        else:
            self.embedding_model = embedding_model
        
        self.vectorstore = None
    
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an image that might be useful for retrieval
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary of metadata
        """
        # Extract basic file metadata
        metadata = {
            "source": image_path,
            "filename": os.path.basename(image_path),
            "file_type": os.path.splitext(image_path)[1][1:].lower(),
        }
        
        # TODO: You can extend this with OCR to extract text from the image
        # For a production system, you might want to use tools like:
        # - Tesseract OCR for general text
        # - Financial-specific OCR tools for extracting numbers, tables
        # - Chart recognition libraries for chart types
        
        return metadata
    
    def create_document_from_image(self, image_path: str) -> Document:
        """
        Create a Document object from an image path
        
        Args:
            image_path: Path to the image
            
        Returns:
            Document object with image path as page_content and metadata
        """
        metadata = self.extract_image_metadata(image_path)
        
        # For images, we'll use the file path as the document content
        # The actual embedding will handle the image processing
        return Document(page_content=image_path, metadata=metadata)
    
    def index_directory(self, directory_path: str) -> FAISS:
        """
        Index all images in a directory
        
        Args:
            directory_path: Path to the directory containing images
            
        Returns:
            FAISS vectorstore with indexed images
        """
        # Find all images in the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, ext)))
            image_paths.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        # Convert images to documents
        documents = [self.create_document_from_image(path) for path in image_paths]
        
        # Create a FAISS vectorstore
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        return self.vectorstore
    
    def save_index(self, save_path: str):
        """Save the FAISS index to disk"""
        if self.vectorstore is not None:
            self.vectorstore.save_local(save_path)
            print(f"Index saved to {save_path}")
        else:
            raise ValueError("No index to save. Please run index_directory first.")
    
    @classmethod
    def load_index(cls, load_path: str, embedding_model: Embeddings = None) -> 'FinancialImageIndexer':
        """Load a FAISS index from disk"""
        if embedding_model is None:
            embedding_model = FinancialMultimodalEmbeddings()
            
        indexer = cls(embedding_model=embedding_model)
        indexer.vectorstore = FAISS.load_local(load_path, embedding_model)
        return indexer
    

class FinancialMultimodalRetriever:
    """Retriever for financial multimodal data"""
    
    def __init__(self, vectorstore: FAISS):
        """
        Initialize the retriever
        
        Args:
            vectorstore: FAISS vectorstore to use for retrieval
        """
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant images for a query
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of Document objects for relevant images
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant images with similarity scores
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples for relevant images
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)


# Example usage
def main():
    # Initialize the image indexer
    indexer = FinancialImageIndexer()
    
    # Index images in a directory
    images_directory = "./financial_images"  # Change this to your images directory
    vectorstore = indexer.index_directory(images_directory)
    
    # Save the index for future use
    indexer.save_index("./financial_images_index")
    
    # Create a retriever
    retriever = FinancialMultimodalRetriever(vectorstore)
    
    # Example queries
    queries = [
        "Show me charts with declining stock prices",
        "Find balance sheets with high debt ratios",
        "Show me income statements with increasing revenue",
        "Find earnings reports with negative growth",
        "Show me charts with bullish patterns",
    ]
    
    # Retrieve relevant images for each query
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve_with_scores(query, k=3)
        
        for doc, score in results:
            print(f"Score: {score:.4f} - {doc.metadata['filename']}")


if __name__ == "__main__":
    main()