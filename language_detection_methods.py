"""
File containing all the methods requesting the Azure service
"""

import os, requests, random, json
import pandas as pd

def getEnvVar():
    """
    Method which returns the subscription key and the url of th Azure service
    Returns:
    subscription_key -- private key used to access the Azure service, picked up from environment variables
    url_service -- URL of the service created in Azure
    """
    subscription_key = os.environ.get('SUBSCRIPTION_KEY')
    url_service = os.environ.get('URL_SERVICE')
    return subscription_key, url_service


def get_several_sentences_detection(sentences_list):
    """
    Method which returns in json format the list of languages of the sentences passed in parameter
    Args:
    sentences_list -- list of sentences
    Returns:
    The response of the Azure service in format Json
    """

    # Retrieving the environment variables
    subscription_key, url_service = getEnvVar()

    # path determining the type of Azure service, here language detection
    path = '/text/analytics/v3.1/languages'
    # full URL of the service  
    constructed_url = url_service  + path

    # list of headers used to call the service
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-type': 'application/json'
    }

    # sentences separation
    liste_phrases = sentences_list.split(",")

    # json serving as body for the request
    liste_json = []

    # for each sentence whose language we want to detect, we add a piece of json
    for i, s in enumerate(liste_phrases):
        liste_json.append({"id" : i, "text" : s})	
    	
    body = {
             "documents": liste_json
           }

    # REST call in POST to the service, retrieval of the response
    response = requests.post(constructed_url, headers=headers, json=body)

    # return of the result formatted in json
    return response.json()



def test_azure_detection(list_of_indexes):
    """
    Method which returns, in json format, the list of languages of sentences whose indexes are passed as parameters
    Args: 
    list_of_indexes -- list of the indexes
    Returns:
    The result of the Azure service with the detected languages
    """

    # file containing the train sentences
    fname = "/home/ubuntu/OCRIA/projet1/flask-cog-services/train_test/x_train.txt"
    list_of_indexes = list_of_indexes.split(",")
    list_of_indexes = [int(s) for s in list_of_indexes]

    # we get the sentences in the file according to their index and we concatenate them with a comma as a separator
    with open(fname, 'r') as f:
        content = f.readlines()
        content_index = [content[i].replace(',',' ').replace('\n','') for i in list_of_indexes]
        concatenated_strings = ",".join(content_index)

    return get_several_sentences_detection(concatenated_strings)
     
     
def test_azure_detection_success(number_of_sentences):
    """
    Method which returns in json format the percentage of correct response from the service for a given number of tests passed as a parameter
    Args: 
    number_of_sentences -- the number of sentences to detect the language
    Returns: 
    the percentage of correct response from the service for a given number of tests passed as a parameter in the Json format
    """

    # file containing the train sentences
    fnameX = "/home/ubuntu/OCRIA/projet1/flask-cog-services/train_test/x_train.txt"
    # file containing the languages corresponding to the train sentences
    fnameY = "/home/ubuntu/OCRIA/projet1/flask-cog-services/train_test/y_train.txt"

    # labels of the codes of the languages in order to match the codes returned by the Azure service
    labels = pd.read_csv("/home/ubuntu/OCRIA/projet1/flask-cog-services/train_test/labels.csv", sep=';')

    # opening the x_train file to get example of sentences 
    with open(fnameX, 'r') as f:
        contentX = f.readlines()
        file_elements_number = len(contentX) 
        list_of_indexes = random.choices(range(file_elements_number), k=int(number_of_sentences))
        content_index = [contentX[i].replace(',',' ').replace('\n','') for i in list_of_indexes]
        concatenated_strings = ",".join(content_index)

    # opening the y_train file to get example of languages 
    with open(fnameY, 'r') as g:
        contentY = g.readlines()

    # results retrieving
    resp = get_several_sentences_detection(concatenated_strings)

    success = 0 # number of good results
    ex_reussite_prediction = ""
    ex_reussite_test = ""
    ex_echec_prediction = ""
    ex_echec_test = ""

    for c, i in enumerate(list_of_indexes):

        iso = resp['documents'][c]['detectedLanguage']['iso6391Name']
        wikicode = labels[labels['Wiki Code'] == iso]

        # sometimes the service can't detect the language
        if iso != '(Unknown)' and wikicode.shape[0] > 0:

            # fetching the good formatted code
            label = wikicode.iloc[0,0]

            # the last good detection is saved for diplay 
            if contentY[i].strip() == label.strip():
                # success, we increase the number of success
                success += 1
                ex_reussite_prediction = label.strip()
                ex_reussite_test = contentY[i].strip()

            # the last bad detection is saved for diplay
            else:
                ex_echec_prediction = label.strip()
                ex_echec_test = contentY[i].strip()

        c = c + 1
 
    # calculating the percentage of success
    result = 100 * success / int(number_of_sentences)
     
    # creation of the dictionary for displaying the results
    dict_result = {}
    dict_result["a- Number of tests"] = int(number_of_sentences)
    dict_result["b- Number of success"] = success
    dict_result["c- Success percentage"] = result
    dict_result["d- Success example"] = {"Predicted language": ex_reussite_prediction, "Real language": ex_reussite_test}
    dict_result["e- Failure example"] = {"Predicted language": ex_echec_prediction, "Real language": ex_echec_test}
     
    #returns the result
    return dict_result
