# --------------------------------------------------------------------------------------------------
# %% import libraries ------------------------------------------------------------------------------
import ast
import neuralcoref
import json
import nltk
from nltk.tree import *
from nltk.corpus import stopwords, treebank
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2
from nltk.corpus import conll2000
from itertools import islice
import io
from stanfordcorenlp import StanfordCoreNLP
# from pycorenlp import StanfordCoreNLP
import stanfordnlp
from stanfordnlp.server import CoreNLPClient
from corenlp_dtree_visualizer.converters import _corenlp_dep_tree_to_spacy_dep_tree
import spacy
from spacy import displacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------------------------------
# %% Special settings & miscellanneous -------------------------------------------------------------
spacy.prefer_gpu()
nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
sp = spacy.load('en_core_web_sm')
# nltk.download('wordnet')

pos_dict = {
# For NLTK pos tagger
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
'PSP' : 'postposition, common in indian langs','DEM' : 'demonstrative, common in indian langs',

# For Spacy POS tagger
'PROPN': 'propper noun, like "apple"','ADP':'conjunction, subordinating or preposition','VERB': 'VERB', 'VBG':'VERB', 'NOUN': 'noun, names of people, places, things, feelings...', '$': 'Symbol', 'NUM':'number', 'SYM': 'Symbol', 'PRON': 'Pronoun, like I, she, you, him ...', 'PUNCT': 'Punctuation', 'X':'email', 'ADJ':'affix','CCONJ':'conjunction, coordinating', 'DET': 'determiner', 'ADJ':'adjective', 'PRON':'pronoun, personal','ADV': 'Adverb', 'PART':'possessive ending', 'AUX':'"be", "do", "have"'
}

dep_dict = {
'ROOT':'root', 'root': 'root', 'dep': 'dependent', 'aux': 'auxiliary', 'aux:pass': 'passive auxiliary', 'cop': 'copula', 'arg': 'argument', 'agent': 'agent', 'comp': 'complement', 'acomp': 'adjectival complement', 'ccomp': 'clausal complement with internal subject', 'xcomp': 'clausal complement with external subject', 'obj': 'object', 'dobj': 'direct object', 'iobj': 'indirect object', 'pobj': 'object of preposition', 'subj': 'subject', 'nsubj': 'nominal subject', 'nsubj:pass': 'passive nominal subject', 'csubj': 'clausal subject', 'csubj:pass': 'passive clausal subject', 'cc': 'coordination', 'conj': 'conjunct', 'expl': 'expletive (expletive \“there\”)', 'mod': 'modifier', 'amod': 'adjectival modifier', 'appos': 'appositional modifier', 'advcl': 'adverbial clause modifier', 'det': 'determiner', 'predet': 'predeterminer', 'preconj': 'preconjunct', 'vmod': 'reduced, non-finite verbal modifier', 'mwe': 'multi-word expression modifier', 'mark': 'marker (word introducing an advcl or ccomp', 'advmod': 'adverbial modifier', 'neg': 'negation modifier', 'rcmod': 'relative clause modifier', 'quantmod': 'quantifier modifier', 'nn': 'noun compound modifier', 'npadvmod': 'noun phrase adverbial modifier', 'tmod': 'temporal modifier', 'num': 'numeric modifier', 'number': 'element of compound number', 'prep': 'prepositional modifier', 'poss': 'possession modifier', 'possessive': 'possessive modifier(s)', 'prt': 'phrasal verb particle', 'parataxis': 'parataxis', 'goeswith': 'goes with', 'punct': 'punctuation', 'ref': 'referent', 'sdep': 'semantic dependent', 'xsubj': 'controlling subject', 'compound': 'compound pair', 'nmod': 'nominal modifier', 'obl': 'oblique nominal', 'case': 'case marking', 'nmod': 'nominal modifier', 'nummod': 'nominal modifier'
}

# --------------------------------------------------------------------------------------------------
# %% Functions -------------------------------------------------------------------------------------

def sent_tokenize_stnlp(paragraph):
    output = []
    doc = nlp(paragraph)
    for i, sentence in enumerate(doc.sentences):
        sent = ' '.join(word.text for word in sentence.words)
        output.append(sent)
    return output


def core_nlp_sentence_split_and_tokenizer(paragraph):
    corenlp_wrapper = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
    ann_sen = corenlp_wrapper.annotate(paragraph, properties={'annotators': 'tokenize, ssplit',
    'pipelineLanguage':'en', 'outputFormat':'json'})
    sentences = []
    for sentence in json.loads(ann_sen)['sentences']:
        temp_sent = []
        for word in sentence['tokens']:
            temp_sent.append(word['word'])
        sentences.append(temp_sent)
    corenlp_wrapper.close()
    return sentences


def remove_puncts(dict_tokenized):
    new_dict = {}
    for key in dict_tokenized:
        sent_list = []
        for sentence in dict_tokenized[key]:
            word_list = []
            for word in sentence:
                if str(word) not in '?,.!;:':
                    word_list.append(word)
            sent_list.append(word_list)
        new_dict[key] = sent_list
    return new_dict


def extract_pos_info(list_with_1_pos_tagged_tokenized_sentence,dict_kind):
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for wrd in list_with_1_pos_tagged_tokenized_sentence:
        if str(wrd[1]) in dict_kind.keys():
            pos_exp = dict_kind[wrd[1]]
        else:
            pos_exp = 'NA'
        parsed_text['word'].append(wrd[0])
        parsed_text['pos'].append(wrd[1])
        parsed_text['exp'].append(pos_exp)
    #return a dataframe of pos and text
    return parsed_text


def convert_parsed_string_to_list(corenlp_syn_parse):
    corenlp_syn_parse = corenlp_syn_parse.replace('(','[')
    corenlp_syn_parse = corenlp_syn_parse.replace(')',']')
    corenlp_syn_parse = corenlp_syn_parse.replace('\n','')
    correct_string = ''
    in_word = False
    for letter in corenlp_syn_parse:
        if letter.lower() in 'qwertyuioplkjhgfdsazxcvbnm,1234567890.' and not in_word:
            correct_string += '"'
            correct_string += letter
            in_word = True
        elif letter.lower() not in 'qwertyuioplkjhgfdsazxcvbnm,1234567890.' and in_word:
            correct_string += '"'
            correct_string += letter
            in_word = False
        else:
            correct_string += letter
    correct_string_2 = ''
    in_empty_space = False
    for letter in correct_string:
        if letter == ' ' and not in_empty_space:
            correct_string_2 += ','
            in_empty_space = True
        elif letter != ' ':
            correct_string_2 += letter
            in_empty_space = False

    return ast.literal_eval(correct_string_2)


def chunk_extractor(synt_parse, chunk):
    chunk = chunk.lower()
    target = synt_parse[synt_parse.lower().find('('+chunk):]
    level = 0
    if chunk.lower() != 'root':
        for i in range(len(target)):
            if target[i] == '(':
                level -= 1
            elif target[i] == ')' and level < 0:
                level += 1
            # print(level, target[i:i+6])
            if level == 0:
                return target[:i+1]
    elif chunk.lower() == 'root':
        return synt_parse


def get_chunk_targets(syn_parse_copy, chunk):
    chunk = chunk.lower()
    syn_parse_copy = syn_parse_copy.replace('\n','')
    targets = []
    while (f'({chunk} ') in syn_parse_copy.lower():
        targets.append(syn_parse_copy[syn_parse_copy.lower().find('('+chunk+' '):])
        syn_parse_copy = syn_parse_copy[syn_parse_copy.lower().find('('+chunk+' ')+2:]
    return targets

# --------------------------------------------------------------------------------------------------
# %% md
# # Data processing part
# %% Extract all data from the text data and make list ---------------------------------------------
filename = '../thesis/thesis_code/training_data'
temp_file = open(filename, 'r').read()
temp_list = temp_file.split('\n')

# %% Make data dict --------------------------------------------------------------------------------
# Remove unnecesarry characters
only_sentences = []
for sentence in temp_list:
    if sentence != '':
        only_sentences.append(sentence)

# --------------------------------------------------------------------------------------------------
# %% Split into single sentences -------------------------------------------------------------------

# ....... with nltk
sentences_nltk = {}
for el in range(0, len(only_sentences), 2):
    sentences_nltk[only_sentences[el]] = sent_tokenize(only_sentences[el+1])

# ....... with stanfordNLP
sentences_STNLP = {}
for el in range(0, len(only_sentences), 2):
    sentences_STNLP[only_sentences[el]] = sent_tokenize_stnlp(only_sentences[el+1])

# ....... with spaCy
sentences_spacy = {}
for el in range(0, len(only_sentences), 2):
    temp_list = []
    for sentence in sp(only_sentences[el+1]).sents:
        temp_list.append(sentence)
    sentences_spacy[only_sentences[el]] = temp_list

# --------------------------------------------------------------------------------------------------
# Analysis of above dicts -----------------------------------------------------------------------
list(sentences_nltk['Dataset_1'])
list(sentences_STNLP['Dataset_1']) # Seems best (clear Segmentation)
list(sentences_spacy['Dataset_1'])

# --------------------------------------------------------------------------------------------------
# %% Tokenize sentences ----------------------------------------------------------------------------

# ....... with nltk
sentences_tokenized_nltk = {}
for key in sentences_STNLP:
    temp_list = []
    for sentence in sentences_STNLP[key]:
        temp_list.append(nltk.word_tokenize(sentence.strip()))
    sentences_tokenized_nltk[key] = temp_list


# ....... with stanfordnlp
sentences_tokenized_stNLP = {}
for el in range(0, len(only_sentences), 2):
    final_list = []
    for i, sentence in enumerate(nlp(only_sentences[el+1]).sentences):
        temp_list = []
        for word in sentence.words:
            temp_list.append(word.text)
        final_list.append(temp_list)
    sentences_tokenized_stNLP[only_sentences[el]] = final_list


# ....... with spacy
sentences_tokenized_spacy = {}
for key in sentences_spacy:
    temp_list = []
    for sentence in sentences_spacy[key]:
        temp_list_2 = []
        for word in sentence:
            temp_list_2.append(word)
        temp_list.append(temp_list_2)
    sentences_tokenized_spacy[key] = temp_list


# ....... with coreNLP
sentences_tokenized_corenlp = {}
for el in range(0, len(only_sentences), 2):
    sentences_tokenized_corenlp[only_sentences[el]] =core_nlp_sentence_split_and_tokenizer(only_sentences[el+1])

# --------------------------------------------------------------------------------------------------
# %% analysis of above dicts -----------------------------------------------------------------------
print(list(sentences_tokenized_nltk['Dataset_1'])[:5])
print(list(sentences_tokenized_stNLP['Dataset_1'])[:5])
print(list(sentences_tokenized_spacy['Dataset_1'])[:5])
print(list(sentences_tokenized_corenlp['Dataset_1'])[:5]) # Doesn't make separation with "-"
    # --> There doesn't seem to be a difference, spacy list is a spacy object btw

# --------------------------------------------------------------------------------------------------
# Removing punctuations ----------------------------------------------------------------------------
sentences_tokenized_nltk_clean = remove_puncts(sentences_tokenized_nltk)
sentences_tokenized_stNLP_clean = remove_puncts(sentences_tokenized_stNLP)
sentences_tokenized_spacy_clean = remove_puncts(sentences_tokenized_spacy)
sentences_tokenized_corenlp_clean = remove_puncts(sentences_tokenized_corenlp)
# --------------------------------------------------------------------------------------------------
# %% POS tagging  ----------------------------------------------------------------------------------

# ....... with NLTK (only here distinction possible between clean and unclean tokenized sentences)
def nltk_POS_tag(tokenized_dict):
    nltk_pos = {}
    for key in tokenized_dict:
        temp_list = []
        for sentence in tokenized_dict[key]:
            temp_list.append(nltk.pos_tag(sentence))
        nltk_pos[key] = temp_list
    return nltk_pos

nltk_pos_default = nltk_POS_tag(sentences_tokenized_nltk)
nltk_pos_clean = nltk_POS_tag(sentences_tokenized_nltk_clean)


# ....... with stanfordnlp
nlp = stanfordnlp.Pipeline()
sentences_POS_stNLP = {}
for el in range(0, len(only_sentences), 2):
    doc = nlp(only_sentences[el+1])
    final_list = []
    for sent in doc.sentences:
        temp_tup_list = []
        for word in sent.words:
            temp_tup_list.append(tuple([word.text, word.pos]))
        final_list.append(temp_tup_list)
    sentences_POS_stNLP[only_sentences[el]] = final_list


# ....... with spacy
sentences_POS_spacy = {}
for key in sentences_spacy:
    final_list = []
    for sentence in sentences_spacy[key]:
        temp_tup_list = []
        for word in sentence:
            temp_tup_list.append(tuple([word, word.pos_]))
        final_list.append(temp_tup_list)
    sentences_POS_spacy[key] = final_list


# ....... with coreNLP
corenlp_wrapper = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
sentences_POS_coreNLP = {}
for key in sentences_STNLP:
    temp_list = []
    for sentence in sentences_STNLP[key]:
        temp_list.append(corenlp_wrapper.pos_tag(sentence))
    sentences_POS_coreNLP[key] = temp_list
corenlp_wrapper.close()

# --------------------------------------------------------------------------------------------------
# %% analysis of POS taggers -----------------------------------------------------------------------
nltk_pos_info = extract_pos_info(nltk_pos_default['Dataset_1'][0],pos_dict)
stNLP_pos_info = extract_pos_info(sentences_POS_stNLP['Dataset_1'][0],pos_dict)
spacy_pos_info = extract_pos_info(sentences_POS_spacy['Dataset_1'][0],pos_dict)
coreNLP_pos_info = extract_pos_info(sentences_POS_coreNLP['Dataset_1'][0],pos_dict)

# Prepare pandas dataframe
only_words = dict()
only_words['word'] = nltk_pos_info['word']
nltk_pos_info['nltk_pos'] = nltk_pos_info['pos']
stNLP_pos_info['stNLP_pos'] = stNLP_pos_info['pos']
spacy_pos_info['spacy_pos'] = spacy_pos_info['pos']
coreNLP_pos_info['coreNLP_pos'] = coreNLP_pos_info['pos']
nltk_pos_info['nltk_ex'] = nltk_pos_info['exp']
stNLP_pos_info['stNLP_ex'] = stNLP_pos_info['exp']
spacy_pos_info['spacy_ex'] = spacy_pos_info['exp']
coreNLP_pos_info['coreNLP_ex'] = coreNLP_pos_info['exp']

df1 = pd.DataFrame(only_words)
df2 = pd.DataFrame(nltk_pos_info)[['nltk_pos','nltk_ex']]    # NLTK sees sandwich as a symbol
df3 = pd.DataFrame(stNLP_pos_info)[['stNLP_pos','stNLP_ex']] # stNLP sees "is" as an aux verb (good)
df4 = pd.DataFrame(spacy_pos_info)[['spacy_pos','spacy_ex']]
df5 = pd.DataFrame(coreNLP_pos_info)[['coreNLP_pos','coreNLP_ex']]

pd.concat([df1,df2,df3,df4,df5], axis=1, sort=False)


# --------------------------------------------------------------------------------------------------
# Removing stop words -- Won't be necessary --------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Lemmatization (From now on only on sentence level, not for dictionaries) -------------------------

# ....... with NLTK

# ....... with stanfordnlp

# ....... with spacy

# --------------------------------------------------------------------------------------------------
# %% Syntactic paring ------------------------------------------------------------------------------

sentence_pos = sentences_POS_stNLP['Dataset_1'][0]
sentence_normal = sentences_STNLP['Dataset_1'][0]

# ....... with NLTK ----> Intersting to build own grammars
grammar1 = r"""NP: {<DT><NN>}
                   {<NNP>+}
                   {<NNP><NN>}
                   {<NN><NNP>}
                   {<NN><NNS>}
                   {<CD><NNS>?<JJ>}
               VP: {<TO><VB><VBN>}
                   {<VBZ>*}
                   {<VBP><VP>}
           """
cp = nltk.RegexpParser(grammar1)
result = cp.parse(sentence_pos)

# ....... with spacy  ----> Will not be very helpful
sentences_spacy['Dataset_1'][0]
spacy_noun_chunks = sentences_spacy['Dataset_1'][0].noun_chunks

# ....... with CoreNLP ----> Gives full parse (syntactic)
nlp_wrapper = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
corenlp_syn_parse = nlp_wrapper.parse(sentence_normal)
nlp_wrapper.close()

syn_list = convert_parsed_string_to_list(corenlp_syn_parse) # Get parsed syntax

chunk_dict = {} # Get desired chunks
for chunk in ['ROOT','S', 'NP','VP', 'ADJP', 'SBAR', 'ADVP']:
    chunks = []
    for target in get_chunk_targets(corenlp_syn_parse,chunk):
        chunks.append(chunk_extractor(target,chunk))
    chunk_dict[chunk] = chunks

layered_chunk_dict = {}
for i in chunk_dict:
    temp_list = []
    for i2 in chunk_dict[i]:
        temp_dict = {}
        temp_output = convert_parsed_string_to_list(i2)
        temp_dict[temp_output[0]] = temp_output[1:]
        temp_list.append(temp_dict)
    layered_chunk_dict[i] = temp_list

# --------------------------------------------------------------------------------------------------
# Inspect all different chunks and their constituent parts here ------------------------------------
display_chunk = 'S'

with pd.option_context('display.width', None, 'display.max_colwidth', -1):
    display(pd.DataFrame(layered_chunk_dict[display_chunk.upper()])) # The amount of this kind of chunks with their contents

with pd.option_context('display.width', None, 'display.max_colwidth', -1):
    display(pd.DataFrame(layered_chunk_dict[display_chunk.upper()][0])) # The constituent parts per chunk occurance

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# %% Dependency paring -----------------------------------------------------------------------------

# ....... with NLTK

# ....... with stanfordnlp
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang='en')
sentence = nlp(sentence_normal)

print(*[f"index: {word.index.rjust(2)}\tword: {word.text.ljust(11)}\tgovernor index: {word.governor}\tgovernor: {(doc.sentences[0].words[word.governor-1].text if word.governor > 0 else 'root').ljust(11)}\tdeprel: {word.dependency_relation}" for word in doc.sentences[0].words], sep='\n') # --> Not that good


# ....... with spacy -------------------------------------------------------------------------------
texts = [only_sentences[1]]
text = texts[0]
doc = sp(sentence_normal)

depparse = {} # Navigating parse tree
text, dep, head_text, head_pos, children = ([] for i in range(5)) # Initialize lists
for token in doc:
    text.append(token.text), dep.append(token.dep_), head_text.append(token.head.text), head_pos.append(token.head.pos_),children.append([child for child in token.children])

temp_list = []
for i in range(len(text)):
    temp_list.append((text[i],dep[i]))
temp_dict = extract_pos_info(temp_list,dep_dict)

depparse['text'] = text
depparse['dep'] = dep
depparse['exp'] = temp_dict['exp']
depparse['head_text'] = head_text
depparse['head_pos'] = head_pos
depparse['children'] = children

df_temp = pd.DataFrame(depparse)
df_temp

# Display the dependency parse
displacy.serve(sentences_spacy['Dataset_1'][0],style="dep")


# ....... with CoreNLP ----> Gives full parse (semantic)

nlp_wrapper = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
corenlp_depparse = nlp_wrapper.annotate(sentence_normal, properties={'annotators': 'depparse', 'outputFormat': 'json'})
nlp_wrapper.close()

workdoc = json.loads(corenlp_depparse)['sentences']
workdoc[0]['basicDependencies']

# --------------------------------------------------------------------------------------------------
# %% Begin "manual" classification -----------------------------------------------------------------
if_then_synonyms = ['if', 'assuming that', 'conceding that', 'granted that', 'in case that', 'on the assumption that', 'supposing that', 'whenever', 'wherever', 'condtion', 'then', 'when', 'unless']

# %% Create dict for pandas framework
pandas_dict = {'Sentence_ID':[], 'Sentence':[], 'Condition_statement':[]}
for i in sentences_STNLP:
    for sentence in sentences_STNLP[i]:
        pandas_dict['Sentence_ID'].append(i),
        pandas_dict['Sentence'].append(sentence),
        condition_present = False
        for word in if_then_synonyms:
            if sentence.lower().find(word) > -1:
                condition_present = True
        if condition_present:
            pandas_dict['Condition_statement'].append(1)
        else:
            pandas_dict['Condition_statement'].append(0)


df = pd.DataFrame(data=pandas_dict)

# --------------------------------------------------------------------------------------------------
# %% Coreference resolution ------------------------------------------------------------------------
# ....... with spacy -------------------------------------------------------------------------------
# neuralcoref.add_to_pipe(sp)
# doc = sp(sentence_normal)
# doc


# --------------------------------------------------------------------------------------------------
# %% Extraction of IF.. THEN information -----------------------------------------------------------

# %% Syntacticly based
layered_chunk_dict
for i in layered_chunk_dict:
    if i == 'SBAR':
        for i2 in layered_chunk_dict[i][0][i]:
            print(i2)

layered_chunk_dict['SBAR'][0]['SBAR']
output_sentence = ''
for i in layered_chunk_dict['SBAR'][0]['SBAR']:
    print(i)
    if type(i[1]) is not type([]):
        output_sentence += i[1]
    elif type(i[1]) is type([]):
        for i2 in i[1]:
            print(i2)

output_sentence
