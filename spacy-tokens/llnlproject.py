# project1
# AUTHOR: Timothy J Naudet

"""

This file is designed to access error-#.json files in the error-analysis repository and the
word2vec meta.json output of the spack-monitor-nlp repository. The accessed error-#.json files
are identical between the two repositories.

"""

import json
import sys
import spacy
import re
from collections import Counter
import glob
import numpy as np
import requests
import llnlproject_github_urls
import time


# Loading a JSON file
def load_json(file):
    f = open(file=file, mode='r')
    # JSON.load() returns the list of dictionaries and reads from the file
    # JSON.loads() returns the dictionary already parsed and reads the strings itself; throws an error for new lines
    j = json.load(f)
    f.close()
    return j


# Print to a JSON file
def write_json(content, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(content, indent=4))
    fd.close()


# Printing to a text file
def print_txt_file(tuple_list_object, filename):
    # NOTE: If the output file is a JSON file, the current glob
    # process throws an error because the output file is not what we wish
    # to use for processing.
    with open(file=filename, mode='w+') as f:
        f.write(f"{'Error_ID'} {'Rank'.ljust(5)} {'Token-Lemma'.ljust(15)} {'Raw Count'}\n")
        for four_item_tup in tuple_list_object:
            f.write(
                f'{str(four_item_tup[0]).ljust(8)} {str(four_item_tup[1]).ljust(5)} {str(four_item_tup[2]).ljust(15)} {str(four_item_tup[3])}\n')
        f.close()

""" SPACY SECTION"""


def split_into_batches(error_text, max_num_words_per_batch):
    words_split_on_spaces = error_text.split(' ')
    num_error_pieces = int(np.ceil(len(words_split_on_spaces) / max_num_words_per_batch))
    batches = []
    for piece_idx in range(num_error_pieces):
        start_idx = piece_idx * max_num_words_per_batch
        end_idx = (piece_idx + 1) * max_num_words_per_batch
        if end_idx > len(words_split_on_spaces):
            end_idx = len(words_split_on_spaces)
        error_piece = ' '.join(words_split_on_spaces[start_idx:end_idx])
        batches.append(error_piece)
    return batches


# Function for JSON File directories
def directory_spacy_tokenization(json_file_directory):
    tokenization_object = Counter()
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    index = 0
    for file in glob.glob(json_file_directory):
        dictionary_list = load_json(file)

        # This immediate below loop is going to get very time-consuming
        for dictionary in dictionary_list:
            pre_context = split_into_batches(dictionary['pre_context'], 1000)
            error_msg = split_into_batches(dictionary['text'], 1000)
            post_context = split_into_batches(dictionary['post_context'], 1000)
            pre_context.extend(error_msg)
            pre_context.extend(post_context)

            # using spacy for the second time to create tokens
            for batch in pre_context:
                parsed_batch = nlp(batch)
                for token in parsed_batch:
                    if re.match('[a-zA-Z]+$', token.lemma_):
                        tokenization_object[token.lemma_] += 1
                index += 1
    return tokenization_object


# Function for SINGLE JSON File Link
def spacy_tokenization(json_file):
    # Instantiate a dictionary to contain all the final id-token pairs for each id
    final_dict = {}

    # Instantiate NLP Object
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    # This immediate below loop is going to get very time-consuming
    for dictionary in json_file:

        # Instantiate counter object
        tokenization_object = Counter()

        pre_context = split_into_batches(dictionary['pre_context'], 10000)
        error_msg = split_into_batches(dictionary['text'], 10000)
        post_context = split_into_batches(dictionary['post_context'], 10000)
        pre_context.extend(error_msg)
        pre_context.extend(post_context)

        # using spacy for the second time to create tokens
        for batch in pre_context:
            parsed_batch = nlp(batch)
            for token in parsed_batch:
                if re.match('[a-zA-Z]+$', token.lemma_):
                    tokenization_object[token.lemma_] += 1

        final_dict[dictionary['id']] = tokenization_object

    return final_dict


# function to sort the tokens
def token_sorting(id_token_object):
    # instantiate list to hold all the tuple objects
    sorted_tuples = []

    # Instantiate loop for each id
    for id_key in id_token_object.keys():

        # Instantiate lists to store objects
        tokens_unsorted = []
        counts_unsorted = []

        # Iterate through the tokens and counts for each histogram
        for token, count in id_token_object[id_key].items():
            tokens_unsorted.append(token)
            counts_unsorted.append(count)

        # convert the list objects into arrays
        counts_unsorted = np.array(counts_unsorted)
        sorted_indices = np.argsort(counts_unsorted)[::-1]

        # Create loop to enumerate through the ranks
        for rank, idx in enumerate(sorted_indices):
            sorted_tuples.append(tuple((id_key, rank + 1, tokens_unsorted[idx], counts_unsorted[idx])))
    return sorted_tuples


# Function for handling URL Requests
def request_url_data(url_param):
    requested_json = requests.get(url_param).json()
    requested_tokens = spacy_tokenization(requested_json)
    requested_tup_list = token_sorting(requested_tokens)
    return requested_tup_list


# Function for multiple URLs
def multiple_raw_urls(url_list):
    url_tuple_list = []
    index = 1
    for each_url in url_list:
        url_tuple_list.extend(request_url_data(each_url))
        print(f"successfully extended {index} urls.")
        print(f"Duration for {index} urls {(time.time() - starttime)/60} minutes ")
        index += 1

    # Returning a list of tuples for each json file and sending this to print
    return url_tuple_list


""" META FILE SECTION """


# Function to split meta.json sections into comparable token objects
def meta_file_splitting(json_dict):
    # instantiate a dictionary for the final object
    final_dict = {}
    print(len(list(json_dict.keys())))
    for key1 in json_dict.keys():
        parsed = []
        counter_object = Counter()
        for key2 in json_dict[key1].keys():
            if "parsed" in key2:
                parsed.extend(json_dict[key1][key2].split(" "))
        for item in parsed:
            if re.match('[a-zA-Z]+$', item):
                counter_object[item] += 1

        final_dict[json_dict[key1]["id"]] = counter_object

    return final_dict

if __name__ == '__main__':
    # Timer Check
    starttime = time.time()

    # System Arguments
    user_name = sys.argv[1]
    repository = sys.argv[2]
    meta_repository = sys.argv[3]

    # Function calls
    # """ GITHUB URLS IMPORTED CODE """
    raw_github_urls = llnlproject_github_urls.git_repos_contents(user_name, repository)

    # Function call to handle multiple raw urls
    print(f"generating data for each of the urls...")
    multiple_raw_url_list = multiple_raw_urls(raw_github_urls)

    # Function call to print the collected data
    print(f"Printing the data for the multiple urls...")
    print_txt_file(multiple_raw_url_list, "json_file_spacy.txt")

    """ META FILE SECTION """
    ### NOTE: The meta.json is a dicionary of dictionaries, not alist like the erro json files.

    meta_url = llnlproject_github_urls.git_repos_contents(user_name, meta_repository)[0]
    meta_json = requests.get(meta_url).json()
    print(type(meta_json))
    print(len(meta_json))
    meta_dict = meta_file_splitting(meta_json)
    meta_tuples = token_sorting(meta_dict)
    print_txt_file(meta_tuples, "json_file_meta.text")

    """ CONFIGURATIONS """
    # python llnlproject.py TimJN17 error_analysis spack-monitor-nlp
    # TimJN17
    # error_analysis
    # spack-monitor-nlp