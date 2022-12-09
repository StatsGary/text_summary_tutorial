""" 
Name:       app.py
Author:     Gary Hutson
Date:       09/12/2022
Usage:      python app.py
"""

from flask import Flask, request, jsonify
import os
# This is the function we are going to use to pass requests to
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
import torch
from datetime import date

MODEL_NAME = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def textsummy(text: str, max_length:int=1024, min_sum_len:int=80, 
                   max_sum_len:int=120, verbose:bool=True):
    # Specify if the text passed is a string
    assert isinstance(text, str), f'Please specify a string input into {text}'
    # Put the model on a GPU for inference
    model.to(device)
    tokens_input = tokenizer.encode('summarize' + text, 
            return_tensors='pt', max_length=max_length, 
            truncation=True)
    tokens_input = tokens_input.to(device)
    ids = model.generate(tokens_input, min_length=min_sum_len, max_length=max_sum_len)
    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    if verbose:
        print(f'Producing summary for text: {text}\nSummary is: {str(summary)}.')

    # Compute metrics
    r = Rouge()
    rouge_scores = r.get_scores(summary, text)
    return summary, rouge_scores



# Create the flask component
app = Flask(__name__)

@app.route('/')
def page_index():
    return jsonify({'status': 'Summarisation model running, ready for inference...',
                    'date': f'{str(date.today())}'
                    })
# POST REQUEST
@app.route('/summarise', methods=['POST'])
def summariser():
    json_request = request.get_json(force=True)
    text = json_request['text']
    min_sum_length = json_request['min_len']
    max_sum_length = json_request['max_len']
    # Use function with JSON passed to get result
    result = textsummy(text=text, 
                   min_sum_len=min_sum_length,
                   max_sum_len=max_sum_length)
    return str(result[0])

if __name__== "__main__":
    app.run(port=int(os.environ.get("PORT", 8081)),
             debug=True)
    #app.run(debug=True)