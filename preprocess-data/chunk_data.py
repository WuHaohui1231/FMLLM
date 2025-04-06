import json
import os
from typing import Iterator, Any
import ijson  # For memory-efficient JSON parsing

def create_output_directory(directory: str) -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def stream_json_list(filename: str) -> Iterator[Any]:
    """
    Stream JSON array items one at a time to avoid loading entire file into memory.
    
    Args:
        filename: Path to the JSON file containing an array of items
        
    Yields:
        Individual items from the JSON array
    """
    with open(filename, 'rb') as file:
        parser = ijson.items(file, 'item')
        yield from parser

def chunk_json_data(input_file: str, output_dir: str, chunk_size: int = 100000) -> None:
    """
    Split a large JSON array file into smaller chunks.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save chunk files
        chunk_size: Number of items per chunk (default: 100000)
    """
    create_output_directory(output_dir)
    
    current_chunk = []
    chunk_number = 0
    
    try:
        for item in stream_json_list(input_file):
            current_chunk.append(item)
            
            # When chunk reaches desired size, write to file
            if len(current_chunk) >= chunk_size:
                output_file = os.path.join(output_dir, f'chunk_{chunk_number}.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(current_chunk, f)
                print(f'Wrote chunk {chunk_number} to {output_file}')
                
                current_chunk = []  # Reset chunk
                chunk_number += 1
        
        # Write remaining items if any
        if current_chunk:
            output_file = os.path.join(output_dir, f'chunk_{chunk_number}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_chunk, f)
            print(f'Wrote final chunk {chunk_number} to {output_file}')
            
    except Exception as e:
        print(f'Error processing file: {str(e)}')
        raise

def main():
    # Configuration
    INPUT_FILE = '/data/hh/FMLLM/data/concatenated_news_data.json'  # Replace with your input file path
    OUTPUT_DIR = '/model/haohui/chunked_financial_news_data_longer_2'  # Output directory
    CHUNK_SIZE = 500000  # Items per chunk
    
    try:
        chunk_json_data(INPUT_FILE, OUTPUT_DIR, CHUNK_SIZE)
        print('Successfully completed chunking process')
    except Exception as e:
        print(f'Failed to process file: {str(e)}')

if __name__ == '__main__':
    main()