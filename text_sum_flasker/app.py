""" 
Name:       sum_flask.py
Author:     Gary Hutson
Date:       29/11/2022
Usage:      python sum_flask.py
            This will create a flask service - please configure URL for different ports

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

def summarise_text(text: str, max_length:int=1024, min_sum_len:int=80, 
                   max_sum_len:int=120, verbose:bool=True):
    '''
    Uses a text summarization model to create a summary of text

            Parameters:
                    text (str): The text to be summarized
                    max_length (int): Optional param (defaults=1024) for the max sequence length
                    min_sum_len (int): Optional param (defaults=80) for the minimum text summary length
                    max_sum_len (int): Optional param (defaults=120) for the maximum summary length 
                    verbose (bool): indicating whether to print out the summary

            Returns:
                    tuple of:
                        summary (str) - the summary generated
                        rouge_scores (dict) - a dictionary of rouge scores
    '''
    if isinstance(text, str):
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
    else:
        raise ValueError('Expecting a string to be passed to the text line')


# Create the flask component
app = Flask(__name__)

@app.route('/')
def page_index():
    return jsonify({'status': 'Summarisation model running, ready for inference...',
                    'date': f'{str(date.today())}'
                    })

@app.route('/welcome', methods=['GET'])
def welcome_page():
    html = """
    <!DOCTYPE html>
    <html>
    <title>Text Summariser Flask App Landing Page</title>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link rel='stylesheet' href='custom.css'>
    <h1>Welcome to the Text Summariser - developed for CRISP</h1>
    <p>To get started with the summariser you need to create a postman request to send a 
    POST request to /summarise.</p>
    <p>This expects the following inputs:</p>
    <ol>
        <li><strong>text</strong> - a JSON text field to summarise</li>
        <li><strong>min_len</strong> - the minimum length of the summary</li>
        <li><strong>max_len</strong> - the maximum length of the summary</li>
    </ol>
    """
    return html

# POST REQUEST
@app.route('/summarise', methods=['POST'])
def summariser():
    json_request = request.get_json(force=True)
    text = json_request['text']
    min_sum_length = json_request['min_len']
    max_sum_length = json_request['max_len']
    # Use function with JSON passed to get result
    result = summarise_text(text=text, 
                   min_sum_len=min_sum_length,
                   max_sum_len=max_sum_length)
    return str(result[0])

if __name__== "__main__":
    # Run in local host 0.0.0.0 on port whatever
    # app.run(port=int(os.environ.get("PORT", 8081)), 
    #         host="0.0.0.0", 
    #         debug=True)
    app.run(debug=True)