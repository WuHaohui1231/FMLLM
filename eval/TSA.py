import json
from transformers import pipeline

data_path = "/model/haohui/FMLLM-Eval/semeval-2017-task-5-subtask-1/Microblog_Trialdata.json"


# Load the data from tsa.json
with open(data_path, 'r') as file:
    data = json.load(file)

# Define the prompt for the model
prompt_template = (
    "Given the following financial text, return a sentiment score for Ashtead as a floating-point number "
    "ranging from -1 (indicating a very negative or bearish sentiment) to 1 (indicating a very positive or bullish sentiment), "
    "with 0 designating neutral sentiment. Return only the numerical score first, follow it with a brief reasoning behind your score.\n\n"
    "Text: {text}"
)

# Function to get sentiment score from the model
# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="/model/haohui/LLaMA-Factory/saves/llama3-2_vision/full/pt_finance_24/checkpoint-2050", tokenizer="/model/haohui/LLaMA-Factory/saves/llama3-2_vision/full/pt_finance_24/checkpoint-2050")

def get_sentiment_score(text):
    prompt = prompt_template.format(text=text)
    result = sentiment_pipeline(prompt)[0]
    score = result['score']
    if result['label'] == 'NEGATIVE':
        score = -score
    return score

# Perform inference and store results
results = []
for entry in data:
    text = " ".join(entry["spans"])
    predicted_score = get_sentiment_score(text)
    entry["model-predicted sentiment score"] = predicted_score
    results.append(entry)

# Save the results to result.json
with open('./result/tsa.json', 'w') as file:
    json.dump(results, file, indent=4)