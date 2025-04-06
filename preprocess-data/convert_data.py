import ijson
import json

def stream_convert_json_format(input_file, output_file):
    # Open output file in write mode
    with open(output_file, 'w') as out_f:
        # Start the JSON array in output file
        out_f.write('[\n')
        
        count = 0
        is_first = True
        
        # Open input file and create parser
        with open(input_file, 'rb') as in_f:
            # Parse the JSON array item by item
            parser = ijson.items(in_f, 'item')
            
            for item in parser:
                count += 1
                
                # Create new format
                new_item = {
                    "text": f"{item['Title']}\n{item['Date']}\n{item['Text']}"
                }
                
                # Add comma for all items except the first
                if not is_first:
                    out_f.write(',\n')
                is_first = False
                
                # Write the item
                json.dump(new_item, out_f)
                
                # Print progress every 5000 items
                if count % 20000 == 0:
                    print(f"Processed {count} items. Last date processed: {item['Date']}")
        
        # Close the JSON array
        out_f.write('\n]')
    
    print(f"Conversion complete. Total items processed: {count}")

# First install ijson if you haven't:
# pip install ijson

# Usage example
if __name__ == "__main__":
    input_file = "/data/hh/FMLLM/partial_data.json"  # Replace with your input file name
    output_file = "partial_concatenated_news_data.json"  # Replace with desired output file name
    stream_convert_json_format(input_file, output_file)