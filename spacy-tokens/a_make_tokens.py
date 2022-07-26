# Coding in utf-8
"""
Created on date: 11 July 2022
@authors: Timothy J. Naudet

Purpose: to write the outbound json file as the same as the incoming json file, but with 3 new fields of tokens for
each technique used.

13 July 2022
Project purpose: To effectively implement SoftNER to determine relevant NE's in the error-*.json
files and compare/vectorize
the error NE's to the StackOverFlow database that the SoftNER authors compiled. Within these comparisons/vectorizations
we can
determine, possibly through vector similarity or sentiment score (with altered labels), a probability of correlation and
recommend these stackoverflow pages when the NE's appear in a produced error message.

"""

from tqdm import tqdm
import argparse
import spacy
import nltk
import re
import time
from river import feature_extraction
from _stokenizer import tokenize
from _utilities import git_raw_urls, request_url_data, write_json, meta_file_splitting, keras_sequencer


# function to perform all the tokenization and addition to the dictionary
def make_tokens(word_list):
    """
    :param word_list: the list of words selected by key selection for each dictionary in the json file
    :return: a tuple of tokens and NER from the appellate techniques
    """

    # spacy section
    spacy_tokens = []
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
    for word in word_list.split(' '):
        if re.search(r'[^./][\w+][/\w+]:', word) and len(word) > 6:
            spacy_tokens.append(word)

    parsed_batch = nlp(word_list)
    for token in parsed_batch:
        if re.match(r'[\w+]+$', token.lemma_):  # [a-zA-Z]+$
            spacy_tokens.append(token.text.lower())
    spacy_tokens = ' '.join(list(set(spacy_tokens)))

    # bagOwords section
    model = feature_extraction.BagOfWords(lowercase=True, strip_accents=True)
    temporary_dict = model.transform_one(word_list)
    bagOwords = list(set(list(temporary_dict.keys())))
    bagOwords = ' '.join(bagOwords)

    # nltk section
    # word = re.compile(r'\w+')
    nltk_tokens = list(set(nltk.word_tokenize(word_list, language='english')))
    for word in word_list.split(' '):
        if re.search(r'[^./][\w+][/\w+]:', word) and len(word) > 6:
            nltk_tokens.append(word)
    nltk_tokens = ' '.join(nltk_tokens)

    # keras section
    keras_tokens = keras_sequencer(word_list)
    for word in word_list.split(' '):
        if re.search(r'[^./][\w+][/\w+]:', word) and len(word) > 6:
            keras_tokens.append(word)
    keras_tokens = list(set(keras_tokens))
    keras_tokens = ' '.join(keras_tokens)

    # stokenizer section
    stokes = word_list.split(' ')
    final_stokes = []
    for word in stokes:
        if re.search(r'[^./][\w+][/\w+]:', word) and len(word) > 6:
            final_stokes.append(word)
            stokes.remove(word)
            if len(stokes) > len(word_list.split(' ')):
                stokes.pop(stokes.index(word))
            else:
                continue

    # must return a line of text (,join) from the list (tokenize) whose parameter is a line of text (.join)
    final_stokes.extend(' '.join(tokenize(' '.join(stokes))))

    for word in final_stokes:
        if not re.match(r'[\w+]+$', word):  # [a-zA-Z]+$
            final_stokes.remove(word)

    return spacy_tokens, bagOwords, nltk_tokens, keras_tokens, final_stokes


# function to produce collect the appropriate words
def words_from_json(incoming_json_text, meta_dictionary):
    """
    :param incoming_json_text: the json that we read in from the github repository; essentially a list of dictionaries,
     but in this case it has been modified to only be the id and the text key, value pairs
    :param meta_dictionary: the entire dictionary from the word2Vec json
    :return: a word list comprising all text in the three mentioned keys
    """

    '''
    NOTE: id 6072 has a significant number of file path / punctuation like strings. This is the problem that
    is causing the significant delays in the program.
    '''

    print(f"\nStart time is: {time.asctime(time.localtime(time.time()))} minutes.")

    final_transformed = {}
    for small_dict in tqdm(incoming_json_text,
                           desc=f"Progress %",
                           position=1,
                           unit=' ID',
                           mininterval=120,
                           miniters=100, leave=True, colour="magenta"):

        index = 0
        word_list = small_dict['text']
        sp, bag, nltk_, keras, stokes = make_tokens(word_list)  #
        small_dict["spacy"] = sp
        small_dict["bagOwords"] = bag
        small_dict["nltk"] = nltk_
        small_dict["keras"] = keras
        small_dict["stoken"] = stokes
        for key in meta_dictionary.keys():
            if key == small_dict['id']:
                small_dict["Doc2Vec"] = ' '.join(list(meta_dictionary[key].keys()))
        final_transformed[small_dict['id']] = small_dict
        index += 1

    return final_transformed


# main function
def main():
    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('User_name', type=str, help='Github repository user_name')
    parser.add_argument('Repository', type=str, help='Repositoqry name')
    args = parser.parse_args()
    user_name = args.User_name
    repository = args.Repository
    raw_urls = git_raw_urls(user_name, repository, "error_files")
    index = 0

    meta_urls = git_raw_urls(user_name=user_name, repository=repository, flag="docs")
    meta_json, _ = request_url_data(meta_urls[0])
    meta_dictionary = meta_file_splitting(meta_json)

    for url in raw_urls[0:2]:
        requested_json, requested_json_text = request_url_data(url)
        outbound_json_text = words_from_json(requested_json_text, meta_dictionary)
        for _dict_ in requested_json:
            for key in outbound_json_text.keys():
                if _dict_['id'] == key:
                    _dict_['spacy'] = outbound_json_text[_dict_['id']]['spacy']
                    _dict_['nltk'] = outbound_json_text[_dict_['id']]['nltk']
                    _dict_['bagOwords'] = outbound_json_text[_dict_['id']]['bagOwords']
                    _dict_['Doc2Vec'] = outbound_json_text[_dict_['id']]['Doc2Vec']
                    _dict_['keras'] = outbound_json_text[_dict_['id']]["keras"]
                    _dict_['stoken'] = outbound_json_text[_dict_['id']]["stoken"]

        write_json(requested_json, f"__TEST_stoken_errors_{index}_tokens.json")
        index += 1


if __name__ == '__main__':
    main()

    # configurations
    # python a_make_tokens.py TimJN17 spack-monitor-nlp
