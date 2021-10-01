from flask import Flask, jsonify, Response

from language_detection_methods import get_several_sentences_detection, test_azure_detection, test_azure_detection_success

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# List of possible routes

# Language detection of sentences passed in parameter

@app.route('/detect_languages/<_text>', methods=['GET'])
def detect_languages(_text):
     return jsonify(get_several_sentences_detection(_text))

# Detection of the languages of  sentences whose indices are passed in parameter

@app.route('/test_languages_list_indices/<_text>', methods=['GET'])
def test_languages(_text):
     return jsonify(test_azure_detection(_text))

# Determine the percentage of good predictions of the service

@app.route('/test_languages_nb_indices/<_nb>', methods=['GET'])
def test_languages_success(_nb):
     return jsonify(test_azure_detection_success(_nb))

