# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet

Purpose: Collect the determined tokens from the make_token.py file; return metrics of comparison between the tokeniztion
techniques; produce a bar chart of the most common tokens

"""

from glob import glob
import numpy as np
from _utilities import read_json, array_writing, write_json


# Function to read the files and create the array
def collect_tokens(incoming_json, final_array, final_json):
    for _dict_ in incoming_json:
        small_dict = {}

        N = [x for x in _dict_['nltk'].split(' ') + _dict_['spacy'].split(' ')
             if x not in _dict_['spacy'].split(' ') and x in _dict_['nltk'].split(' ')]
        B = [x for x in _dict_['bagOwords'].split(' ') + _dict_['spacy'].split(' ')
             if x not in _dict_['spacy'].split(' ') and x in _dict_['bagOwords'].split(' ')]
        D = [x for x in _dict_['Doc2Vec'].split(' ') + _dict_['spacy'].split(' ')
             if x not in _dict_['spacy'].split(' ') and x in _dict_['Doc2Vec'].split(' ')]
        K = [x for x in _dict_['keras'].split(' ') + _dict_['spacy'].split(' ')
             if x not in _dict_['spacy'].split(' ') and x in _dict_['keras'].split(' ')]
        S = [x for x in _dict_['stoken'].split(' ') + _dict_['spacy'].split(' ')
             if x not in _dict_['spacy'].split(' ') and x in _dict_['stoken'].split(' ')]

        small_dict['id'] = _dict_['id']
        small_dict['text'] = _dict_['text']
        small_dict['spacy'] = _dict_['spacy']
        small_dict['nltk'] = _dict_['nltk']
        small_dict['bagOwords'] = _dict_['bagOwords']
        small_dict['Doc2Vec'] = _dict_['Doc2Vec']
        small_dict['keras'] = _dict_['keras']
        small_dict['stoken'] = _dict_['stoken']

        sl = len(_dict_['spacy'].split(' '))

        metrics = [round(((sl - len(N)) / sl) * 100, 2),
                   round(((sl - len(B)) / sl) * 100, 2),
                   round(((sl - len(D)) / sl) * 100, 2),
                   round(((sl - len(K)) / sl) * 100, 2),
                   round(((sl - len(S)) / sl) * 100, 2)]

        final_array = np.vstack((final_array,
                                 np.array([_dict_['id'], metrics[0], metrics[1], metrics[2],
                                           metrics[3], metrics[4], _dict_['text']])))

        small_dict['Spacy-nltk'] = metrics[0]
        small_dict['Spacy-BagOWords'] = metrics[1]
        small_dict['Spacy-Doc2Vec'] = metrics[2]
        small_dict['Spacy-Keras'] = metrics[3]
        small_dict['Spacy-Stokenizer'] = metrics[4]

        final_json.append(small_dict)

    final_array = np.delete(final_array, obj=0, axis=0)

    return final_array, final_json


# Main Function
def main():
    final_array = np.ndarray(shape=(1, 7))
    final_json = []

    for file in glob("_stoken_?_tokens.json"):
        list_of_dicts = read_json(file)

        outbound_array, outbound_json = collect_tokens(list_of_dicts, final_array=final_array, final_json=final_json)

        final_array = np.vstack((final_array, outbound_array))
        final_json.extend(outbound_json)

    final_metric = [round(float(np.mean([final_array[:, 1]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 2]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 3]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 4]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 5]], dtype=np.float64)), 2)]

    header = f"The average Percentages of overlap with the Spacy Tokens is for each column are: \n " \
             f"\t\t\t\t {final_metric[0]} \t\t  {final_metric[1]} \t\t  {final_metric[2]} \t\t  {final_metric[3]} \t\t  {final_metric[4]}\n" \
             f"The columns represent the percentage of tokens are common with the spacy tokens\n" \
             f"{'Error_ID  '.ljust(7)}   {'Spacy-NLTK'.ljust(7)}   {'Sp-BagOWs'.ljust(7)}  " \
             f"{'Sp-Doc2V '.ljust(7)}   {'Spcy-Keras'.ljust(7)}   {'Sp-Stoknzr'.ljust(7)}   " \
             f"{'Error_Text'.ljust(7)} "

    final_array = np.delete(final_array, obj=0, axis=0)

    final_array = np.vstack((np.array([['AVG VALS:', str(final_metric[0]), str(final_metric[1]),
                                        str(final_metric[2]), str(final_metric[3]), str(final_metric[4]), ' ']]),
                             final_array))

    final_json.insert(0, {"id": "AVG VALS",
                          "Spacy-nltk": float(final_metric[0]),
                          "Spacy-BagOWords": float(final_metric[1]),
                          "Spacy-Doc2Vec": float(final_metric[2]),
                          "Spacy-Keras": float(final_metric[3]),
                          "Spacy-Stoken": float(final_metric[4])})

    array_writing(final_array, header=header, filename="_presenting_array.txt")
    write_json(content=final_json, filename="_presenting_json.json")


if __name__ == '__main__':
    main()

    # Configurations
    # python 2.compare_token_metrics.py _stoken_0_tokens.json
