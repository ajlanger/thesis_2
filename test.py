# --------------------------------------------------------------------------------------------------
# %% import libraries ------------------------------------------------------------------------------
import json
import nltk
from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP
import spacy
from spacy import displacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------------------------------
# %% Other configurations --------------------------------------------------------------------------
spacy.prefer_gpu()

# --------------------------------------------------------------------------------------------------
# %% Simple example CORENLP ------------------------------------------------------------------------
nlp = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
print('Tokenize:', nlp.word_tokenize(sentence))
print('Part of Speech:', nlp.pos_tag(sentence))
print('Named Entities:', nlp.ner(sentence))
print('Constituency Parsing:', nlp.parse(sentence))
print('Dependency Parsing:', nlp.dependency_parse(sentence))
nlp.close() # Do not forget to close! The backend server will consume a lot memory.
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# %% Extract all data from the text data and make library ------------------------------------------
filename = '../thesis/thesis_code/data.txt'

temp_file = open(filename, 'r').read()
temp_list = temp_file.split('\n')

# %% Make data dict --------------------------------------------------------------------------------
only_sentences = []
for sentence in temp_list:
    if sentence != '':
        only_sentences.append(sentence)

temp_dict_1 = {}
for el in range(0, len(only_sentences), 2):
    temp_dict_1[only_sentences[el]] = only_sentences[el+1].split('.')[:-1]

data_dict = {}
for i in temp_dict_1:
    temp_list_2 = []
    for sentence in temp_dict_1[i]:
        temp_list_2.append(sentence.strip()+'.')
    data_dict[i] = temp_list_2
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# %% Tokenize and POS-tag sentences ----------------------------------------------------------------
sentence_list = [data_dict['Dataset_1'][0]]

# With NLTK
tokenized_data = []
for sentence in sentence_list:
    tokenized_data.append(nltk.word_tokenize(sentence))

tagged_data = []
for sentence in tokenized_data:
    tagged_data.append(nltk.pos_tag(sentence))

# With StanfordCore NLP
sentence = sentence_list[0]

nlp = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
output = nlp.pos_tag(sentence)
nlp.close()

# --------------------------------------------------------------------------------------------------
# %% Give syntactic and semantic parse -------------------------------------------------------------
nlp = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
# constituency_output = nlp.parse(sentence)
overall_output = nlp.annotate(sentence,
                              properties={'annotators':'parse, depparse',
                              'outputformat':'json'})
nlp.close()
overall_output = json.loads(overall_output)
# %%
to_tree = [overall_output['sentences'][0]['parse']]
Tree.fromstring(to_tree[0])
overall_output['sentences'][0]['basicDependencies']

# --------------------------------------------------------------------------------------------------
# %% Spacy -----------------------------------------------------------------------------------------
space = spacy.load('en_core_web_sm')
doc = space('This is a sentence.')

# displacy.render(doc, style="dep")
