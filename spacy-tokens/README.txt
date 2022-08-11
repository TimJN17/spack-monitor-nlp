###Tokenization Process	

##Author: Timothy Naudet

# End Result: 
A json file containing the error ID, the text, the tokens produces by teqhcniques involving spacy, nltk, bagOwords, Doc2Vec, keras, and stokenizer, as well as the percentages that each tokenizer overlaps with the spacy tokens. 

# List of executable scripts:
1.make_github_tokens.py
2.compare_token_metrics.py
3.plot_token_counts.py
4.tfidf.py

# Approach:
The process includes called **1.make_github_tokens.py** to access the “spack-monitor-nlp” repository, use the errors-*.json raw urls, and collect the text from each entry in the files. This script then returns two json files named “_stoken_*_tokens.json”. An import anecdote is that regex expressions were implemented to capture the file paths provided in the error messages and stored within the tokenization component of each json entry. These files are then used in 2.compare_token_metrics.py.

**2.compare_token_metrics.py** uses the glob package to open each of the aforementioned stoken json files and calculate the percentage of overlap of each tokenization technique with the spacy tokenization technique. This percentage is a basic comparison: 

The metric is calculated as a percentage of the number of 
non-spacy tokens in the non-spacy technique:

((number_spacy_tokens – number_not_in_spacy)/number_spacy_tokens) * 100

These metrics are the provide for each non-spacy tokenization technique in the singular outbound json file.

**3.plot_token_counts.py** plots two seaborn bar-charts in one image. These list the 30 most common tokens for all tokens as well as the 30 most common file paths (separated using regex). 

**4.tfidf.py** prepare a few deeper analyses of the “error-?.json” text files. 
