# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet

Purpose: Perform spacy tokenization. These will e compared to word2vec and bagOwords as well as the '2005' paper SoftNER

"""

from _utilities import split_into_batches, git_raw_urls, request_url_data, print_txt_file, token_sorting
import argparse
import spacy
import re
from collections import Counter
import numpy as np
import time


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
                    """Possible alterations for the below line"""
                    tokenization_object[token.text.lower()] += 1

        final_dict[dictionary['id']] = tokenization_object

    return final_dict


# function to sort the tokens
# def token_sorting(id_token_object):
#     # instantiate list to hold all the tuple objects
#     sorted_tuples = []
#
#     # Instantiate loop for each id
#     for id_key in id_token_object.keys():
#
#         # Instantiate lists to store objects
#         tokens_unsorted = []
#         counts_unsorted = []
#
#         # Iterate through the tokens and counts for each histogram
#         for token, count in id_token_object[id_key].items():
#             tokens_unsorted.append(token)
#             counts_unsorted.append(count)
#
#         # convert the list objects into arrays
#         counts_unsorted = np.array(counts_unsorted)
#         sorted_indices = np.argsort(counts_unsorted)[::-1]
#
#         # Create loop to enumerate through the ranks
#         for rank, idx in enumerate(sorted_indices):
#             sorted_tuples.append(tuple((id_key, rank + 1, tokens_unsorted[idx], counts_unsorted[idx])))
#     return sorted_tuples

def main():
    starttime = time.time()
    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('User_name', type=str, help='Github repository user_name')
    parser.add_argument('Repository', type=str, help='Repository name')
    args = parser.parse_args()
    user_name = args.User_name
    repository = args.Repository
    raw_urls = git_raw_urls(user_name, repository, "error_files")
    final_tuple_list = []
    index = 1
    for url in raw_urls:
        requested_json = request_url_data(url)
        requested_tokens = spacy_tokenization(requested_json)
        request_tuple_list = token_sorting(requested_tokens)
        final_tuple_list.extend(request_tuple_list)
        print(f"successfully extended {index} urls.")
        print(f"Duration for {index} urls {(time.time() - starttime) / 60} minutes ")
        index += 1
    print_txt_file(final_tuple_list, "_1.spacy_tokens.txt")


if __name__ == '__main__':
    main()

    # configurations
    # python 1.spacy.py TimJN17 spack-monitor-nlp
