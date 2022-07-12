# Coding in utf-8
"""
Created on date: 11 July 2022
@authors: Timothy J. Naudet

Purpose: to write the outbound json file as the same as the incoming json file, but with 3 new fields of tokens for
each technique used.

"""

import argparse
import time
import spacy
import nltk
import re
from river import feature_extraction
from _utilities import split_into_batches, git_raw_urls, request_url_data, write_json, meta_file_splitting


# function to perform all the tokenization and addition to the dictionary
def make_tokens(word_list):
    """
    :param word_list: the list of words selected by key selection for each dictionary in the json file
    :return: a tuple of tokens and NER from the appellate techniques
    """
    # spacy section
    spacy_tokens = []
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    batches = split_into_batches(word_list, 10000)
    for batch in batches:
        parsed_batch = nlp(batch)
        for token in parsed_batch:
            if re.match(r'[a-zA-Z]+$', token.lemma_):
                spacy_tokens.append(token.text.lower())
    spacy_tokens = ' '.join(spacy_tokens)
    print(f"Spacy tokens complete.")

    # bagOwords section
    model = feature_extraction.BagOfWords(lowercase=True, strip_accents=True)
    temporary_dict = model.transform_one(word_list)
    bagOwords = list(temporary_dict.keys())
    bagOwords = ' '.join(bagOwords)
    print(f"BagOWords tokens complete.")

    # nltk section
    nltk_tokens = nltk.word_tokenize(word_list)
    nltk_tokens = ' '.join(nltk_tokens)
    print(f"NLTK tokens complete.")

    return spacy_tokens, bagOwords, nltk_tokens


# function to produce collect the appropriate words
def words_from_json(incoming_json, meta_dictionary):
    """
    :param incoming_json: the json that we read in from the github repository; essentially a list of dictionaries
    :param meta_dictionary: the entire dictionary from the word2Vec json
    :return: a word list comprising all text in the three mentioned keys
    """
    final_transformed = []
    for small_dict in incoming_json:
        print(f"=======Current id is: {small_dict['id']}=========")
        word_list = small_dict['text']
        word_list += small_dict['pre_context']
        word_list += small_dict['post_context']
        small_dict["spacy"] = make_tokens(word_list)[0]
        small_dict["bagOwords"] = make_tokens(word_list)[1]
        small_dict["nltk"] = make_tokens(word_list)[2]
        for key in meta_dictionary.keys():
            if key == small_dict['id']:
                small_dict["word2vec"] = ' '.join(list(meta_dictionary[key].keys()))
        final_transformed.append(small_dict)

    return final_transformed


# main function
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
    index = 0

    meta_urls = git_raw_urls(user_name=user_name, repository=repository, flag="docs")
    meta_json = request_url_data(meta_urls[0])
    meta_dictionary = meta_file_splitting(meta_json)

    for url in raw_urls:
        requested_json = request_url_data(url)
        outbound_json = words_from_json(requested_json[0:500], meta_dictionary)
        write_json(outbound_json, f"_errors_{str(index)}_tokens.json")
        index += 1
        print(f"successfully extended {index} urls.")
        print(f"Duration for {index} urls {(time.time() - starttime) / 60} minutes ")


if __name__ == '__main__':
    main()

    # configurations
    # python 4.token_comparison.py TimJN17 spack-monitor-nlp
