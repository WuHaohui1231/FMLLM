# Multimodal LLMs for Heterogeneous Financial Data Understanding and Question Answering with Retrieval Augmented Generation

## Overview

We propose to train an MLLM for fine-grained understanding of multimodal financial data. Firstly, we continually pretrain on the MLLM with an extensive financial news corpus to enrich its domain knowledge. Then, we conduct supervised instruction finetuning with a combination of finance QA datasets. Evaluation on (financial) document QA benchmarks demonstrates notable improvement of our model over the foundation Llama 3.2 Vision. Additionally, performing financial tasks sometimes requires out-of-context or the latest knowledge. To handle this scenario, we integrate our model with retrieval augmented generation (RAG). We propose a novel RAG framework transforming images into text descriptions for embedding, which overcomes the issue of loss of fine-grained information during embedding in other RAG methods, and provides a unified method of information retrieval over heterogeneous modalities.  Our evaluation shows our RAG’s superior retrieval accuracy and significant effectiveness in augmenting MLLMs for out-of-context tasks.

## Usage

### Getting Started

```
conda create -n "FMLLM" python=3.12
conda activate FMLLM
pip install -r requirements.txt
```

### Example
```
python example.py --data path_to_input_data --output_dir directory_to_store_result
```

## Key Features

- **Domain-Specific Understanding**: Enhanced capabilities for interpreting financial language and complex text-visual layouts in financial documents
- **Improved Performance**: Significant gains over foundation models on financial document QA benchmarks
- **Novel RAG Framework**: Transforms images into text descriptions for embedding, enabling unified information retrieval across heterogeneous modalities
- **Knowledge Beyond Input Context**: Integrates with retrieval augmentation to access the latest financial information not present in the model's training data



## Architecture

Our approach consists of two main components:

1. **Domain-Specialized MLLM**:
   - Continual pretraining on financial corpus
   - Supervised instruction finetuning with finance QA datasets
   - Built on Llama 3.2 Vision foundation

2. **Multimodal RAG Framework**:
   - Converts image content to text descriptions
   - Performs unified embedding across modalities
   - Preserves fine-grained information during embedding
   - Enables accurate retrieval of relevant information

## Results

Our evaluation demonstrates:
- Superior performance on financial document QA benchmarks compared to the foundation model
- Improved retrieval accuracy through our novel RAG approach
- Effective handling of out-of-context financial tasks requiring the latest information



## Citation



## License



## Acknowledgements

