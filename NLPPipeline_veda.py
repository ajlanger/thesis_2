# %% markdown
# Main TEX2DEC NLP Pipeline with the following stages
#
# Goal: Text(business description) to XML(DMN) is the goal
# input: Text
# output: XML
# %% markdown
# Uses NLTK, StanfordNLP, stanfordCoreNLP, Spacy and other libs
# %% codecell
#read input text
import re
import json

import pandas as pd

def read_file(filename):
    with open(filename, 'r' ,encoding="utf8") as file:
         text = file.read()
    return text

text = "Whenever it 's Monday , and the weather is rainy , vegetable sandwiches are to be made ."

print('Original texual description: %s' % (text))
print()
# %% markdown
# #Step 1: Sentence Segmentation
# #Step 2: Word Tokenization
# #Step 3: Predicting Parts of Speech for Each Token
# #Step 4: Text Lemmatization
# #Step 5: Identifying Stop Words
# #Step 6: Dependency Parsing
# #Step 6b: Finding Noun Phrases
# #Step 7: Named Entity Recognition (NER)
# #Step 8: Coreference Resolution
# #Step 9: Anaphora Resolution
# #step 10: Tuple generation (XML format)
# %% codecell
#import nltk pacakge
import nltk

print('NTLK version: %s' % (nltk.__version__))
# %% codecell
#import spacy pipeline
import spacy

print('Spacy version: %s' % (spacy.__version__))
# %% codecell
sp = spacy.load('en_core_web_sm') #initialising spacy pipeline small
ssentence = sp(text)
# %% codecell
#importing stanfordnlp package
import torch
import stanfordnlp
import pytorch

print('stanfordnlp version: %s' % (stanfordnlp.__version__))

# %% codecell
stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()#default stanfordnlp pipeline
doc = nlp(text)

# %% codecell
#importing stanfordCoreNLP package

from stanfordcorenlp import StanfordCoreNLP
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
# %% markdown
# Step 1: Sentence Segmentation
# The first step in the pipeline is to break the text apart into separate sentences.
# %% codecell
from nltk import sent_tokenize
#Breaking text into sentences using sentence tokenization from NLTK


sentences = sent_tokenize(text)
print(sentences)

# %% codecell
#Breaking text into sentences using stanfordnlp pipeline
#doc.sentences[0].text

for i, sentence in enumerate(doc.sentences):
    sent = ' '.join(word.text for word in sentence.words)
    print(sent)
# %% codecell
#Breaking text into sentences using spacy pipeline

document = sp(text)
for sentence in document.sents:
    print(sentence)
# %% codecell
#Breaking text into sentences using stanfordCorenlp pipeline

des = "Ronaldo has moved from Real Madrid to Juventus. While messi still plays for Barcelona"
annot_doc = nlp_wrapper.annotate(des,
    properties={
        'annotators': 'ner, pos, depparse',
        'outputFormat': 'json',
        'timeout': 1000,
    })

print(json.dumps(annot_doc, indent=2))

# %% markdown
# Step 2: Word Tokenization
# %% codecell
#breaking text into words using Word tokenization from NLTK

words = nltk.word_tokenize(text)
print(words)
# %% codecell
#breaking text into words using stanfordnlp

#print(type(doc.sentences[0]))
print([word.text for sent in doc.sentences for word in sent.words])
#print(*[f'text: {word.text+" "}\tlemma: {word.lemma}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')
# %% codecell
#tokenisation using spacy

#for word in ssentence:
#    print(word.text)

# Extract tokens for the given doc
print ([token.text for token in ssentence])
# %% codecell
#removing the punctuations
punctuations="?:!.,;"

for word in words:
    if word in punctuations:
        words.remove(word)
print(words)
# %% markdown
# Step 3: Predicting Parts of Speech for Each Token
# %% codecell
#dictionary that contains pos tags and their explanations
pos_dict = {
'CC': 'coordinating conjunction','CD': 'cardinal digit','DT': 'determiner',
'EX': 'existential there (like: \"there is\" ... think of it like \"there exists\")',
'FW': 'foreign word','IN':  'preposition/subordinating conjunction','JJ': 'adjective \'big\'',
'JJR': 'adjective, comparative \'bigger\'','JJS': 'adjective, superlative \'biggest\'',
'LS': 'list marker 1)','MD': 'modal could, will','NN': 'noun, singular \'desk\'',
'NNS': 'noun plural \'desks\'','NNP': 'proper noun, singular \'Harrison\'',
'NNPS': 'proper noun, plural \'Americans\'','PDT': 'predeterminer \'all the kids\'',
'POS': 'possessive ending parent\'s','PRP': 'personal pronoun I, he, she',
'PRP$': 'possessive pronoun my, his, hers','RB': 'adverb very, silently,',
'RBR': 'adverb, comparative better','RBS': 'adverb, superlative best',
'RP': 'particle give up','TO': 'to go \'to\' the store.','UH': 'interjection errrrrrrrm',
'VB': 'verb, base form take','VBD': 'verb, past tense took',
'VBG': 'verb, gerund/present participle taking','VBN': 'verb, past participle taken',
'VBP': 'verb, sing. present, non-3d take','VBZ': 'verb, 3rd person sing. present takes',
'WDT': 'wh-determiner which','WP': 'wh-pronoun who, what','WP$': 'possessive wh-pronoun whose',
'WRB': 'wh-abverb where, when','QF' : 'quantifier, bahut, thoda, kam (Hindi)','VM' : 'main verb',
'PSP' : 'postposition, common in indian langs','DEM' : 'demonstrative, common in indian langs'
}
# %% codecell
#pos tags using NLTK
poswords = nltk.pos_tag(words)
print(poswords)

#extract parts of speech
def extract_pos(poswords):
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for wrd in poswords:
        if wrd[1] in pos_dict.keys():
            pos_exp = pos_dict[wrd[1]]
        else:
            pos_exp = 'NA'
        parsed_text['word'].append(wrd[0])
        parsed_text['pos'].append(wrd[1])
        parsed_text['exp'].append(pos_exp)
    #return a dataframe of pos and text
    return pd.DataFrame(parsed_text)

#extract pos
extract_pos(poswords)
# %% codecell
#pos tags using Stanfordnlp pipeline
#extract parts of speech
def extract_pos(doc):
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            if wrd.pos in pos_dict.keys():
                pos_exp = pos_dict[wrd.pos]
            else:
                pos_exp = 'NA'
            parsed_text['word'].append(wrd.text)
            parsed_text['pos'].append(wrd.pos)
            parsed_text['exp'].append(pos_exp)
    #return a dataframe of pos and text
    return pd.DataFrame(parsed_text)

#extract pos
extract_pos(doc)
# %% codecell
#pos tags using Spacy pipeline
#for word in ssentence:
#    print(word.text,  word.pos_)

#extract parts of speech
def extract_pos(ssentence):
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for wrd in ssentence:
        if wrd.pos_ in pos_dict.keys():
            pos_exp = pos_dict[wrd.pos_]
        else:
            pos_exp = 'NA'
        parsed_text['word'].append(wrd.text)
        parsed_text['pos'].append(wrd.pos_)
        parsed_text['exp'].append(pos_exp)
    #return a dataframe of pos and text
    return pd.DataFrame(parsed_text)

#extract pos
extract_pos(ssentence)
# %% markdown
# Step 4: Text Lemmatization (Very optional)
# %% codecell
#word lemmatization using nltk
nltk.download('wordnet')
# %% codecell

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#words
#print("{0:20}{1:20}".format("Word","Lemma"))
#for word in words:
#    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))

parsed_text = {'word':[], 'lemma':[]}
for wrd in words:
    #extract text and lemma
    parsed_text['word'].append(wrd)
    parsed_text['lemma'].append(wordnet_lemmatizer.lemmatize(wrd))

df = pd.DataFrame(parsed_text)
df
# %% codecell
#word lemmatisation using stanfordnlp

#extract lemma
def extract_lemma(doc):
    parsed_text = {'word':[], 'lemma':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            #extract text and lemma
            parsed_text['word'].append(wrd.text)
            parsed_text['lemma'].append(wrd.lemma)
    #return a dataframe
    return pd.DataFrame(parsed_text)

#call the function on doc
extract_lemma(doc)

# %% codecell
#word lemmatisation using spacy

#for word in ssentence:
#    print(word.text+"\t"+word.lemma_)

parsed_text = {'word':[], 'lemma':[]}
for wrd in ssentence:
    #extract text and lemma
    parsed_text['word'].append(wrd.text)
    parsed_text['lemma'].append(wrd.lemma_)

df = pd.DataFrame(parsed_text)
df
# %% markdown
# Step 5: Identifying Stop Words
# %% codecell
nltk.download('stopwords')
# %% codecell
import io
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
for r in words:
    if not r in stop_words:
        appendFile = open('filteredtext.txt','a')
        appendFile.write(" "+r)
        appendFile.close()
# %% markdown
# Step 6: Dependency Parsing (using nltk, standofordnlp, spacy)
# %% codecell
#dependency parsing using NLTK

#Step 6b identifying the noun phrases
#ideally tags should be unified ('The', 'DT'), ('Risk', 'NNP'), ('level', 'NN') into a single NP
#"NP: {<DT>?<JJ>*<NN>}
#                  {<NN><NN>}  "

#applying pattern
pattern = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
import nltk
cp = nltk.RegexpParser(pattern)
cs = cp.parse(poswords)
print(cs)
# %% codecell
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)
# %% codecell

# %% codecell
dep_dict = {
'root': 'root',
'dep': 'dependent',
'aux': 'auxiliary',
'aux:pass': 'passive auxiliary',
'cop': 'copula',
'arg': 'argument',
'agent': 'agent',
'comp': 'complement',
'acomp': 'adjectival complement',
'ccomp': 'clausal complement with internal subject',
'xcomp': 'clausal complement with external subject',
'obj': 'object',
'dobj': 'direct object',
'iobj': 'indirect object',
'pobj': 'object of preposition',
'subj': 'subject',
'nsubj': 'nominal subject',
'nsubj:pass': 'passive nominal subject',
'csubj': 'clausal subject',
'csubj:pass': 'passive clausal subject',
'cc': 'coordination',
'conj': 'conjunct',
'expl': 'expletive (expletive \“there\”)',
'mod': 'modifier',
'amod': 'adjectival modifier',
'appos': 'appositional modifier',
'advcl': 'adverbial clause modifier',
'det': 'determiner',
'predet': 'predeterminer',
'preconj': 'preconjunct',
'vmod': 'reduced, non-finite verbal modifier',
'mwe': 'multi-word expression modifier',
'mark': 'marker (word introducing an advcl or ccomp',
'advmod': 'adverbial modifier',
'neg': 'negation modifier',
'rcmod': 'relative clause modifier',
'quantmod': 'quantifier modifier',
'nn': 'noun compound modifier',
'npadvmod': 'noun phrase adverbial modifier',
'tmod': 'temporal modifier',
'num': 'numeric modifier',
'number': 'element of compound number',
'prep': 'prepositional modifier',
'poss': 'possession modifier',
'possessive': 'possessive modifier(s)',
'prt': 'phrasal verb particle',
'parataxis': 'parataxis',
'goeswith': 'goes with',
'punct': 'punctuation',
'ref': 'referent',
'sdep': 'semantic dependent',
'xsubj': 'controlling subject',
'compound': 'compound pair',
'nmod': 'nominal modifier',
'obl': 'oblique nominal',
'case': 'case marking'
}
# %% codecell
#dependency parsing using stanford nlp
for sent in doc.sentences:
    print("------------------")
    sent.print_dependencies()
#print(type(doc.sentences[0].print_dependencies()))
# %% codecell
#tuples
#print(doc.sentences[0].dependencies[0][0].text)
# %% codecell
#dependency_parse = doc.sentences[0].basicDependencies
#print(dependency_parse)
# %% codecell
#extract and explain dependencies
def extract_pos(doc):
    parsed_text = {'word1':[],'word2':[], 'dep':[], 'exp':[]}
    for sent in doc.sentences:
        for dep in sent.dependencies:
            if dep[1] in dep_dict.keys():
                dep_exp = dep_dict[dep[1]]
            else:
                dep_exp = 'NA'
            parsed_text['word1'].append(dep[0].text)
            parsed_text['word2'].append(dep[2].text)
            parsed_text['dep'].append(dep[1])
            parsed_text['exp'].append(dep_exp)
    #return a dataframe of pos and text
    return pd.DataFrame(parsed_text)

#extract pos
extract_pos(doc)
# %% codecell
#identifying compound nouns for optimal dependencies using stanford nltk

adj_doc = nlp("The company's customer service was terrible.")
verb_doc = nlp("They kept increasing my phone bill")

def get_compound_pairs(doc, verbose=False):
    """Return tuples of (multi-noun word, adjective or verb) for document."""
    compounds = [tok for tok in doc if tok.dep_ == 'compound'] # Get list of compounds in doc
    compounds = [c for c in compounds if c.i == 0 or doc[c.i - 1].dep_ != 'compound'] # Remove middle parts of compound nouns, but avoid index errors
    tuple_list = []
    if compounds:
        for tok in compounds:
            pair_item_1, pair_item_2 = (False, False) # initialize false variables
            noun = doc[tok.i: tok.head.i + 1]
            pair_item_1 = noun
            # If noun is in the subject, we may be looking for adjective in predicate
            # In simple cases, this would mean that the noun shares a head with the adjective
            if noun.root.dep_ == 'nsubj':
                adj_list = [r for r in noun.root.head.rights if r.pos_ == 'ADJ']
                if adj_list:
                    pair_item_2 = adj_list[0]
                if verbose == True: # For trying different dependency tree parsing rules
                    print("Noun: ", noun)
                    print("Noun root: ", noun.root)
                    print("Noun root head: ", noun.root.head)
                    print("Noun root head rights: ", [r for r in noun.root.head.rights if r.pos_ == 'ADJ'])
            if noun.root.dep_ == 'dobj':
                verb_ancestor_list = [a for a in noun.root.ancestors if a.pos_ == 'VERB']
                if verb_ancestor_list:
                    pair_item_2 = verb_ancestor_list[0]
                if verbose == True: # For trying different dependency tree parsing rules
                    print("Noun: ", noun)
                    print("Noun root: ", noun.root)
                    print("Noun root head: ", noun.root.head)
                    print("Noun root head verb ancestors: ", [a for a in noun.root.ancestors if a.pos_ == 'VERB'])
            if pair_item_1 and pair_item_2:
                tuple_list.append((pair_item_1, pair_item_2))
    return tuple_list

get_compound_pairs(adj_doc)
#>>>[(customer service, terrible)]
get_compound_pairs(verb_doc)
#>>>[(phone bill, increasing)]
#get_compound_pairs(example_doc, verbose=True)
#>>>Noun:  compound dependency
#>>>Noun root:  dependency
#>>>Noun root head:  identifies
#>>>Noun root head rights:  []
#>>>Noun:  compound nouns
#>>>Noun root:  nouns
#>>>Noun root head:  identifies
#>>>Noun root head verb ancestors:  [identifies]
#>>>[(compound nouns, identifies)]
# %% markdown
# Visualize the Dependency tree
# %% codecell
#Visualise dependency tree with nltk

import nltk
nltk.download('treebank')
# %% codecell
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()
# %% codecell

# %% codecell
# Load the large English NLP model
#nlp = spacy.load('en_core_web_lg')

# The text we want to examine
#text = """London is the capital and most populous city of England and
#the United Kingdom.  Standing on the River Thames in the south east
#of the island of Great Britain, London has been a major settlement
#for two millennia. It was founded by the Romans, who named it Londinium.
#"""
# %% codecell
# Parse the text with spaCy. This runs the entire pipeline.
document = nlp(text)
displacy.serve(document, style="dep")
# could also save to a file
svg = displacy.render(doc, style="dep",jupyter=False)
with open('tmp.svg', 'w', encoding='utf-8') as fw:
    fw.write(svg)
# %% markdown
# Step 7 named entity extraction
# %% codecell
#Named entity extraction using spacy

displacy.render(document, style='ent', jupyter=True)

# 'doc' now contains a parsed version of text. We can use it to do anything we want!
# For example, this will print out all the named entities that were detected:
for entity in document.ents:
    print(f"{entity.text} ({entity.label_})")
# %% codecell
#scrubber function using named entities using stanfordnlp

# Replace a token with "REDACTED" if it is a name
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == "PERSON":
        return "[REDACTED] "
    else:
        return token.string

# Loop through all the entities in a document and check if they are names
def scrub(text):
    doc = nlp(text)
    for ent in doc.ents:
        ent.merge()
    tokens = map(replace_name_with_placeholder, doc)
    return "".join(tokens)

s = """
In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence". In 1957, Noam Chomsky’s
Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.
"""

print(scrub(s))
# %% markdown
# #Convert Stanford CoreNLP's Dependency Tree to Spacy's for Visualization
# %% codecell
import spacy
from spacy import displacy
from stanfordnlp.server import CoreNLPClient

from corenlp_dtree_visualizer.converters import _corenlp_dep_tree_to_spacy_dep_tree


# Input text
print(text)

# Get a dependency tree from a Stanford CoreNLP pipeline
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse'],
        timeout=60000, memory='8G', output_format='json') as client:
    # submit the request to the server
    ann = client.annotate(text)

# Convert dependency tree formats
sent = ann['sentences'][0]
tree = _corenlp_dep_tree_to_spacy_dep_tree(sent['tokens'], sent['enhancedPlusPlusDependencies'])

# Visualize with Spacy
nlp = spacy.load("en_core_web_sm")
displacy.render(tree, style="dep", manual=True)

# could also save to a file
svg = displacy.render(tree, style="dep", manual=True)
with open('tmp.svg', 'w', encoding='utf-8') as fw:
 fw.write(svg)
# %% codecell
#StanfordNLP's Dependency Tree to Spacy's for Visualization
import os
from nltk.parse.stanford import StanfordDependencyParser
from graphviz import Source

# make sure nltk can find stanford-parser
# please check your stanford-parser version from brew output (in my case 3.6.0)
os.environ['CLASSPATH'] = r'/usr/local/Cellar/stanford-parser/3.6.0/libexec'

sentence = 'The brown fox is quick and he is jumping over the lazy dog'

sdp = StanfordDependencyParser()
result = list(sdp.raw_parse(sentence))

dep_tree_dot_repr = [parse for parse in result][0].to_dot()
source = Source(dep_tree_dot_repr, filename="dep_tree", format="png")
source.view()
# %% markdown
# Step 8 Anaphora resolution and coreference resolution
# %% markdown
# Step 8 Tuples to XML (DMN compatible format)
