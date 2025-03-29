import os
import json

data_dir = "/model/junfeng/GraphRAG-DataSet/news/data"
output_file = "consolidated_financial_news_data.json"
# output_file = "partial_financial_news_data.json"

def combine_json_files(data_dir, output_file):
    consolidated_data = []

    for year in sorted(os.listdir(data_dir)):
    # for year in ["2022"]:
        year_path = os.path.join(data_dir, year)

        if os.path.isdir(year_path):
            print(f"Processing year: {year}")

            for json_file in sorted(os.listdir(year_path)):
                if json_file.endswith(".json"):
                    file_path = os.path.join(year_path, json_file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            daily_data = json.load(f)
                            consolidated_data.extend(daily_data)

                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

    print(f"Consolidation complete. Output written to {output_file}")

if __name__ == "__main__":
    combine_json_files(data_dir, output_file)
