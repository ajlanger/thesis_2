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


# ███████ ██    ██ ██████  ██████   ██████  ██████  ████████
# ██      ██    ██ ██   ██ ██   ██ ██    ██ ██   ██    ██
# ███████ ██    ██ ██████  ██████  ██    ██ ██████     ██
#      ██ ██    ██ ██      ██      ██    ██ ██   ██    ██
# ███████  ██████  ██      ██       ██████  ██   ██    ██



# ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
# █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
# ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████


def get_texts(filename):
    filename = filename
    temp_file = open(filename, 'r').read()
    temp_list = temp_file.split('\n')
    return temp_list


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


def extract_words(target):
    if nill(target):
        return []
    elif len(target) == 1 and type(target[0]) is type([]):
        return extract_words(target[0])
    elif len(target) == 2 and type(target[0]) is type('str') and type(target[1]) is type('str'):
        return target[1]
    elif type(target[0]) is type([]) and type(target[1]) is type([]):
        return extract_words(car(target)), extract_words(cdr(target))
    elif type(target[0]) is type('str') and type(target[1]) is type([]):
        return extract_words(cdr(target))


def clean_statement(statement):
    statement = str(extract_words(statement))
    statement = statement.replace('(', '')
    statement = statement.replace(')', '')
    statement = statement.replace("'", '')
    statement = statement.replace(',', '')
    return statement


def car(a_list):
    return a_list[0]


def cdr(a_list):
    return a_list[1:]


def nill(a_list):
    return len(a_list) == 0


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def output_empty(output):
    for key in output:
        if output[key] == []:
            return True
    else:
        return False


def make_string(words_list):
    return sp(' '.join([word.text for word in words_list]))


def get_dep_parse(sentence):
    if 'spacy' not in str(type(sentence)).lower():
        sentence = sp(sentence)
    return displacy.render(sentence,style="dep")


def get_token_children(token):
    if token.is_sent_start:
        return token
    else:
        return token, [get_token_children(child) for child in token.children]


def remove_duplicate_chunks(words_list):
    temp_index_list = []
    output = []
    for word in words_list:
        temp_index_list.append(word.idx)
    idx_uniques = list(dict.fromkeys(temp_index_list))
    for i in words_list:
        if i.idx in idx_uniques:
            output.append(i)
            idx_uniques.remove(i.idx)
    return output


def get_distinct_sentences(cond,cons):
    condition_sentence = ''
    consequence_sentence = ''
    for word in cond:
        condition_sentence += word.text + ' '
    for word in cons:
        consequence_sentence += word.text + ' '
    if condition_sentence in consequence_sentence:
        # Shorten consequence part
        cons = remove_wrong_part(cond, cons)
    elif consequence_sentence in condition_sentence:
        # Shorten condition part
        cond = remove_wrong_part(cons, cond)
    return [cond, cons]


def get_distinct_sentences_v2(cond,cons):
    cond_idx = [word.idx for word in cond]
    cons_idx = [word.idx for word in cons]
    pop_list = []
    for index in cons_idx:
        if index in cond_idx:
            pop_list.append(cons_idx.index(index))
    remove_elements(cons, pop_list)
    return [cond, cons]


def remove_wrong_part(companion, list_to_shorten):
    if 'list' not in str(type(list_to_shorten)):
        list_to_shorten = get_tokens_spacy(list_to_shorten)
    if 'list' not in str(type(companion)):
        companion = get_tokens_spacy(companion)
    companion_idx = [i.idx for i in companion]
    pop_list = []
    for i in range(0, len(list_to_shorten)):
        if list_to_shorten[i].idx in companion_idx:
            pop_list.append(i)

    # Last new part that was added to avoid error
    # Original:
    """""""""""
    def remove_wrong_part(companion, list_to_shorten):
        companion_idx = [i.idx for i in companion]
        pop_list = []
        for i in range(0, len(list_to_shorten)):
            if list_to_shorten[i].idx in companion_idx:
                pop_list.append(i)

        for i in sorted(pop_list, reverse=True):
            del list_to_shorten[i]
        return list_to_shorten
    """""""""""
    for i in sorted(pop_list, reverse=True):
        del list_to_shorten[i]
    return list_to_shorten


def get_only_sentences(temp_list):
    only_sentences = []
    for sentence in temp_list:
        if sentence != '':
            only_sentences.append(sentence)
    return only_sentences


def get_pos_tags_spacy(sentence):
    if 'spacy' not in str(type(sentence)).lower():
        sentence = sp(sentence)
    return [(word.text, word.pos_) for word in sentence]


def get_tokens_spacy(sentence):
    if 'spacy' not in str(type(sentence)):
        sentence = sp(sentence)
    return [word for word in sentence]


def nltk_POS_tag(tokenized_dict):
    nltk_pos = {}
    for key in tokenized_dict:
        temp_list = []
        for sentence in tokenized_dict[key]:
            temp_list.append(nltk.pos_tag(sentence))
        nltk_pos[key] = temp_list
    return nltk_pos


def get_spacy_lib(only_sentences):
    sentences_spacy = {}
    for el in range(0, len(only_sentences), 2):
        temp_list = []
        for sentence in sp(only_sentences[el+1]).sents:
            temp_list.append(sentence)
        sentences_spacy[only_sentences[el]] = temp_list
    return sentences_spacy


def get_dep_df(doc):
    # Navigating parse tree
    depparse = {}
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
    return df_temp


def remove_elements(remove_list, indexlist):
    """
    Remove a range of items from a list at once using the indexes.
    """
    for i in sorted(indexlist, reverse=True):
        del remove_list[i]
    return remove_list

def get_root(doc):
    return [el for el in doc if el.dep_=='ROOT'][0]


def make_dict(list):
    output_dict = {}
    for el in list:
        output_dict[el] = []
    return output_dict


def dict_empty(dictio):
    for key in dictio:
        if dictio[key] != []:
            return False
    return True


def get_subtree(doc, tag):
    for word in doc:
        if word.dep_ == tag:
            return [w for w in word.subtree]

#  ██████  ██████  ███    ██ ██████  ██ ████████ ██  ██████  ███    ██
# ██      ██    ██ ████   ██ ██   ██ ██    ██    ██ ██    ██ ████   ██
# ██      ██    ██ ██ ██  ██ ██   ██ ██    ██    ██ ██    ██ ██ ██  ██
# ██      ██    ██ ██  ██ ██ ██   ██ ██    ██    ██ ██    ██ ██  ██ ██
#  ██████  ██████  ██   ████ ██████  ██    ██    ██  ██████  ██   ████

# ███████ ██   ██ ████████ ██████   █████   ██████ ████████  ██████  ██████  ███████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██    ██ ██   ██ ██
# █████     ███      ██    ██████  ███████ ██         ██    ██    ██ ██████  ███████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██    ██ ██   ██      ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██     ██████  ██   ██ ███████

# ██   ██ ██  ██████  ██   ██     ██      ███████ ██    ██ ███████ ██
# ██   ██ ██ ██       ██   ██     ██      ██      ██    ██ ██      ██
# ███████ ██ ██   ███ ███████     ██      █████   ██    ██ █████   ██
# ██   ██ ██ ██    ██ ██   ██     ██      ██       ██  ██  ██      ██
# ██   ██ ██  ██████  ██   ██     ███████ ███████   ████   ███████ ███████

def implied_condition_consequence_extractor(doc):
    """
    Function to extract condition-consequence from sentence without actual condition indicator
    """
    root_word = get_root(doc)
    before = []
    after = []
    consequence = []

    for w in root_word.children:
        if w.dep_ == 'nsubj' or w.dep_ == 'nsubjpass':
            condition = [wi for wi in w.subtree]
        else:
            if w.idx < root_word.idx:
                # Append everything before root
                [before.append(i) for i in w.subtree]
            else:
                # Append everything after root
                [after.append(i) for i in w.subtree]
    for i in before:
        consequence.append(i)
    consequence.append(root_word)
    for i in after:
        consequence.append(i)

    return {'condition': condition, 'consequence': consequence}


def condition_consequence_extractor_v3(doc):
    if condition_identifier(doc):
        possible_conditions, possible_ors = get_possible_conditions(doc)
        condition_part, split_key = get_condition_v3(possible_conditions)
        consequence_part = get_consequence_v3(doc, split_key)
        # Perform cleaning on output:
        output = {'condition': remove_duplicate_chunks(condition_part), 'consequence': remove_duplicate_chunks(consequence_part)}
        disctinct_sentences = get_distinct_sentences(condition_part, consequence_part)
        output = {'condition': disctinct_sentences[0], 'consequence': disctinct_sentences[1]}

        return output
    else:
        return 'No conditional statement in sentence'


def get_other_part(doc, first_part):
    doc_idx = [word.idx for word in doc]
    first_part_idx = [word.idx for word in first_part]
    output = []
    for word in doc:
        if word.idx not in first_part_idx:
            output.append(word)
    return output


def condition_consequence_extractor_v4(doc):
    if condition_identifier(doc):
        doc_dep_tags = [word.dep_ for word in doc]
        possible_conditions, possible_ors = get_possible_conditions(doc)
        # If the root indicates the consequence
        if not dict_empty(possible_conditions) and 'relcl' not in doc_dep_tags:
            condition_part, split_keys = get_condition_v4(doc, possible_conditions, possible_ors)
            consequence_part = get_root_subtree_without_tagx(doc, split_keys)
        else:
            # If root indicates the condition
            consequence_part = get_subtree(doc, 'relcl')
            condition_part = get_other_part(doc, consequence_part)
        ############################################################################################
        # Probably not needed anymore
        # Perform cleaning on output:
        output = {'condition': remove_duplicate_chunks(condition_part), 'consequence': remove_duplicate_chunks(consequence_part)}
        disctinct_sentences = get_distinct_sentences_v2(condition_part, consequence_part)
        output = {'condition': disctinct_sentences[0], 'consequence': disctinct_sentences[1]}

        return output
    else:
        print('No conditional statement in sentence')
        return implied_condition_consequence_extractor(doc)


def get_condition_v3(possible_conditions):
    for key in possible_conditions:
        if possible_conditions[key] != []:
            condition = flatten(possible_conditions[key])
            return condition, key


def get_condition_v4(doc, possible_conditions, possible_ors):
    doc_idx = [word.idx for word in doc]
    keys = []
    key_length = []
    for key in possible_conditions:
        if possible_conditions[key] != []:
            keys.append(key)
            key_length.append(len(possible_conditions[key][0]))

    # Take the condition with highest len
    key = keys[key_length.index(max(key_length))]

    # However, if the advcl key is not empty, take advcl as key
    if 'advcl' in keys:
        key = 'advcl'
    elif 'prep' in keys:
        key = 'prep'

    possible_conditions_idx = [word.idx for word in possible_conditions[key][0]]
    condition = flatten(possible_conditions[key])

    if possible_ors['conj'] != []:
        possible_ors_idx = [word.idx for word in possible_ors['conj'][0]]
        # If the first conj word comes right after the condition part, append it

        if (doc_idx.index(possible_conditions[key][0][-1].idx) - doc_idx.index(possible_ors['conj'][0][0].idx) == -1):
            for word in possible_ors['conj'][0]:
                condition.append(word)
            return condition, [key, 'conj']
    return condition, [key]

def get_consequence_v3(doc, split_key):
    consequence = []
    root_word = get_root(doc)

    before =[]
    after = []
    for child in root_word.children:
        if child.dep_ != split_key:
            if child.idx < root_word.idx:
                # Append everything before root
                [before.append(i) for i in child.subtree]
            else:
                # Append everything after root
                [after.append(i) for i in child.subtree]
    for i in before:
        consequence.append(i)
    consequence.append(root_word)
    for i in after:
        consequence.append(i)
    return consequence


def get_root_subtree_without_tagx(doc, split_keys):
    consequence = []
    root_word = get_root(doc)

    before =[]
    after = []
    for child in root_word.children:
        # If the tag is not in split_keys and theres no condition in that subtree, append it to cons
        if child.dep_ not in split_keys or not condition_identifier(' '.join([c.text for c in child.subtree])):
            if child.idx < root_word.idx:
                # Append everything before root
                [before.append(i) for i in child.subtree]
            else:
                # Append everything after root
                [after.append(i) for i in child.subtree]
    for i in before:
        consequence.append(i)
    consequence.append(root_word)
    for i in after:
        consequence.append(i)
    return consequence


def get_possible_conditions(doc):
    """
    Get all possible conditions using the [advcl, ccomp, xcomp, prep or conj] tags using the condition_identifier
    """
    dep_tags_split = ['advcl', 'ccomp', 'xcomp', 'prep', 'conj']
    output_dict_conditions = {'advcl':[], 'prep':[], 'xcomp':[], 'ccomp':[], 'conj':[]}
    output_dict_conditions_elses = {'advcl':[], 'prep':[], 'xcomp':[], 'ccomp':[], 'conj':[]}
    consequence = []
    dep_tag_words = [word for word in doc if word.dep_ in dep_tags_split]
    doc_idx = [wi.idx for wi in doc]
    for word in doc:
        # Get all different condition possibilities
        if word.dep_ in dep_tags_split:
            string = ''
            for w in [subtreeword for subtreeword in word.subtree]:
                string += w.text + ' '
            if condition_identifier(string):
                output_dict_conditions[word.dep_].append([subtreeword for subtreeword in word.subtree])
            # It's possible that the previous word is not accounted for, which could result in a false for the condition_identifier
            elif not condition_identifier(string):
                string = [w for w in word.ancestors][0].text + ' '
                for w in [subtreeword for subtreeword in word.subtree]:
                    string += w.text + ' '
                if condition_identifier(string):
                    output_dict_conditions[word.dep_].append([subtreeword for subtreeword in word.subtree])
                #output_dict_conditions[word.dep_].append([subtreeword for subtreeword in word.subtree])
            #elif else_identifier(string):
            #    output_dict_conditions_elses[word.dep_].append([subtreeword for subtreeword in word.subtree])
    # Append only the conjs originating from root, because those are more likely to be dislocated from the condition part and have more risk of being added to the consequence
    for child in get_root(doc).children:
        if child.dep_ == 'conj':
            output_dict_conditions_elses[child.dep_].append([c for c in child.subtree])
            # Also append any cc words that were left behind
            for words in doc:
                if words.dep_ == 'cc' and (doc_idx.index(words.idx) - doc_idx.index(output_dict_conditions_elses[child.dep_][0][0].idx) == -1):
                    output_dict_conditions_elses[child.dep_][0].insert(0, words)

    for word in doc:
        append_conj_to_output = True
        if word.dep_ == 'conj' and not condition_identifier(string):
            # Get dislocated conjunction parts too (not in the condition part)
            list_to_append = [subtreeword for subtreeword in word.subtree]

            # Check whether none of the words appear in other already found parts
            found_condition_idx_list = []
            for key in output_dict_conditions_elses:
                for element in output_dict_conditions_elses[key]:
                    for wordidx in element:
                        found_condition_idx_list.append(wordidx.idx)
            for el in list_to_append:
                if el.idx in found_condition_idx_list:
                    append_conj_to_output = False
            # Also append any adjoining ands or ors or advmods
            if append_conj_to_output:
                for words in doc:
                    if words.dep_ == 'cc' and (doc_idx.index(words.idx) - doc_idx.index(list_to_append[0].idx) == -1):
                        list_to_append.insert(0, words)
                output_dict_conditions_elses[word.dep_].append(list_to_append)
    return output_dict_conditions, output_dict_conditions_elses


def get_possible_subtrees(doc, tag_list = ['advcl', 'ccomp', 'xcomp', 'prep', 'conj']):
    """
    Get all possible conditions using the [advcl, ccomp, xcomp, prep or conj] tags using the condition_identifier
    """
    dep_tags_split = tag_list
    output_dict_conditions = make_dict(tag_list)
    output_dict_conditions_elses = make_dict(tag_list)
    consequence = []
    dep_tag_words = [word for word in doc if word.dep_ in dep_tags_split]
    doc_idx = [wi.idx for wi in doc]
    for word in doc:
        # Get all different condition possibilities
        if word.dep_ in dep_tags_split:
            string = ''
            for w in [subtreeword for subtreeword in word.subtree]:
                string += w.text + ' '
            if condition_identifier(string):
                output_dict_conditions[word.dep_].append([subtreeword for subtreeword in word.subtree])
            # It's possible that the previous word is not accounted for, which could result in a false for the condition_identifier
            elif not condition_identifier(string):
                string = [w for w in word.ancestors][0].text + ' '
                for w in [subtreeword for subtreeword in word.subtree]:
                    string += w.text + ' '
                if condition_identifier(string):
                    output_dict_conditions[word.dep_].append([subtreeword for subtreeword in word.subtree])
                #output_dict_conditions[word.dep_].append([subtreeword for subtreeword in word.subtree])
            #elif else_identifier(string):
            #    output_dict_conditions_elses[word.dep_].append([subtreeword for subtreeword in word.subtree])
    # Append only the conjs originating from root, because those are more likely to be dislocated from the condition part and have more risk of being added to the consequence
    for child in get_root(doc).children:
        if child.dep_ == 'conj':
            output_dict_conditions_elses[child.dep_].append([c for c in child.subtree])
            # Also append any cc words that were left behind
            for words in doc:
                if words.dep_ == 'cc' and (doc_idx.index(words.idx) - doc_idx.index(output_dict_conditions_elses[child.dep_][0][0].idx) == -1):
                    output_dict_conditions_elses[child.dep_][0].insert(0, words)

    for word in doc:
        append_conj_to_output = True
        if word.dep_ == 'conj' and not condition_identifier(string):
            # Get dislocated conjunction parts too (not in the condition part)
            list_to_append = [subtreeword for subtreeword in word.subtree]

            # Check whether none of the words appear in other already found parts
            found_condition_idx_list = []
            for key in output_dict_conditions_elses:
                for element in output_dict_conditions_elses[key]:
                    for wordidx in element:
                        found_condition_idx_list.append(wordidx.idx)
            for el in list_to_append:
                if el.idx in found_condition_idx_list:
                    append_conj_to_output = False
            # Also append any adjoining ands or ors or advmods
            if append_conj_to_output:
                for words in doc:
                    if words.dep_ == 'cc' and (doc_idx.index(words.idx) - doc_idx.index(list_to_append[0].idx) == -1):
                        list_to_append.insert(0, words)
                output_dict_conditions_elses[word.dep_].append(list_to_append)
    return output_dict_conditions, output_dict_conditions_elses


def condition_identifier(sentence):
    # if_then_synonyms_words = ['if', 'whenever', 'wherever', 'then', 'when', 'unless']
    # if_then_synonyms_phrase = ['assuming that ', 'conceding that ', 'granted that ', 'in case that ', 'on the assumption that ', 'supposing that ', 'in case of ', 'in the case of ', 'in the case that ']

    # Tokenize sentence:
    if 'spacy' in str(type(sentence)).lower():
        sentence = sentence.text

    if_then_synonyms_words = ['if', 'whenever', 'wherever', 'when', 'unless', 'presuming']
    if_then_synonyms_phrases = ['in the case that','assuming that', 'conceding that ', 'granted that', 'in case that', 'on the assumption that', 'supposing that ', 'in case of ', 'in the case of ', 'in the case that', 'on condition that ', 'on the condition that', 'given that', 'if and only if ', 'presuming that', 'providing that', 'provided that', 'contingent on ', 'whenever that', 'in the event that']

    # Check words

    for sentence_word in nltk.word_tokenize(sentence):
        if sentence_word.lower() in if_then_synonyms_words:
            return True
        else:
            for wordphrase in if_then_synonyms_phrases:
                if wordphrase in sentence.lower():
                    return True
    return False


def else_identifier(sentence):
    else_synonyms_words = ['differently', 'otherwise', 'diversely', 'contrarily', 'elseways', 'else']
    else_synonyms_phrases = ['any other way ', 'if not ', 'in different circumstances ', 'on the other hand ', 'or else ', 'or then ']

    # Check words
    for sentence_word in nltk.word_tokenize(sentence):
        if sentence_word.lower() in else_synonyms_words:
            return True, sentence_word
        else:
            for wordphrase in else_synonyms_phrases:
                if wordphrase in sentence.lower():
                    return True, wordphrase
    return False, None


####################################################################################################
# OLD FUNCTIONS ####################################################################################

def condition_consequence_extractor(doc):
    """""""""
    Temporary solution for problem of inaccurate cond-cons prediction
    """""""""
    #doc = sp(lookup_replace_if_synonyms(str(doc)))
    """""""""
    Temporary solution for problem of inaccurate cond-cons prediction
    """""""""
    # First check whether there is a conditional statement in the sentence
    if condition_identifier(doc.text):
        # try:
        #     output = extract_condition_consequence_2(doc)
        #     print(1)
        # except:
        output = extract_condition_consequence_1(doc)
        print(2)
        # Check if none of the outputs is empty, if so search for other completion
        if output_empty(output):
            output = extract_condition_consequence_3(doc)
            print(3)
        # Last part added to check whether cond and cons are together in output (bad)
        #if get_distinct_sentences(output['condition'], output['consequence']):
        #    output = extract_condition_consequence_4(doc)
        output = {'condition': remove_duplicate_chunks(output['condition']), 'consequence': remove_duplicate_chunks(output['consequence'])}
        disctinct_sentences = get_distinct_sentences(output['condition'], output['consequence'])
        output = {'condition': disctinct_sentences[0], 'consequence': disctinct_sentences[1]}
        return output
    else:
        return 'No conditional statement in sentence'


def extract_condition_consequence_1(doc):
    condition = []
    consequence = []
    possible_conditions, possible_ors = get_possible_conditions(doc)
    for key in possible_conditions:
        if possible_conditions[key] != []:
            for token in doc:
                if token.dep_ == key:
                    # Previously: [condition.append(child) for child in token.subtree]
                    condition = flatten(possible_conditions[key])
                if token.dep_ == 'ROOT':
                    before =[]
                    after = []
                    for child in token.children:
                        if child.dep_ != key:
                            if child.idx < token.idx:
                                # Append everything before root
                                [before.append(i) for i in child.subtree]
                            else:
                                # Append everything after root
                                [after.append(i) for i in child.subtree]
                    for i in before:
                        consequence.append(i)
                    consequence.append(token)
                    for i in after:
                        consequence.append(i)
            return {'condition': condition, 'consequence': consequence}


def extract_condition_consequence_2(doc):
    consequence = []
    condition = []
    for token in doc:
        if token.dep_ == 'advcl':
            [condition.append(child) for child in token.subtree]
    # find the child with the outcoming advcl relation
    for token in doc:
        for child in token.children:
            if child.dep_ == 'advcl':
                advcl_parent = token
    advcl_parent_children = []
    for child in advcl_parent.children:
        if child.dep_ != 'advcl':
            [advcl_parent_children.append(sub) for sub in child.subtree]
    temp_output = []
    advcl_parent_parents = []
    for parent in advcl_parent.ancestors:
        advcl_parent_parents.append(parent)
        for sub in parent.children:
            if sub != advcl_parent:
                temp_output.append(get_token_children(sub))
    full_string = []
    for el in temp_output[0]:
        if type(el) is list:
            for sublist in el:
                full_string = [sublist] + full_string
        else:
            full_string = [el] + full_string
    garbage = []
    garbage.append(advcl_parent)
    garbage.extend(advcl_parent_parents)
    garbage.extend(full_string)
    garbage.extend(advcl_parent_children)

    # setting the consequence in good order
    consequence_numbers = []
    for word in garbage:
        consequence_numbers.append(word.idx)
    consequence_numbers.sort()
    for i in consequence_numbers:
        for word in garbage:
            if i == word.idx:
                consequence.append(word)

    return {'condition': condition, 'consequence': consequence}

# Drastic times call for drastic measures, I want to replace all IF synonyms in a sentence to If. This is because its synonyms often cause problems. Namely, the synonyms don't have one simple pos tag or dependency tag.

def lookup_replace_if_synonyms(text):
    if 'spacy' in str(type(text)):
        text = str(text)
    if_then_synonyms_words = ['if', 'whenever', 'wherever', 'then', 'when', 'unless']
    if_then_synonyms_phrase = ['in the case that ','assuming that ', 'conceding that ', 'granted that ', 'in case that ', 'on the assumption that ', 'supposing that ', 'in case of ', 'in the case of ', 'in the case that ', 'on condition that ', 'on the condition that ', 'given that ', 'if and only if ', 'presuming that ', 'presuming ', 'providing that ', 'provided that ', 'contingent on ', 'whenever that ', 'in the event that ']
    for sentence_word in nltk.word_tokenize(text):
        for wordphrase in if_then_synonyms_phrase:
            if wordphrase in text.lower():
                indices_to_replace = find_word_indices_string(text, wordphrase)
                return replace_sentence_part_by_index(text, indices_to_replace)
            else:
                if sentence_word.lower() in if_then_synonyms_words:
                    indices_to_replace = find_word_indices_string(text, sentence_word)
                    return replace_sentence_part_by_index(text, indices_to_replace)
    return text


def find_word_indices_string(sentence, word):
    return sentence.lower().find(word), sentence.lower().find(word)+len(word)


def replace_sentence_part_by_index(text, indices_to_replace):
    # If first is 0, means word is in beginning of sentence
    if indices_to_replace[0] == 0:
        return 'If ' + text[indices_to_replace[1]:]
    else:
        return text[:indices_to_replace[0]] + 'if ' + text[indices_to_replace[1]:]

# Functions that would be good for some sentences

# For the tuscany sandwiches sentence --> In order to have the "and" part in the cond instead of cons
def cut_change_if_needed(original_doc, cond, cons):
    original_idx = [sword.idx for sword in original_doc]
    cons_idx = [word.idx for word in cons]
    cond_idx = [word.idx for word in cond]

    cons_chron = [original_idx.index(idx_num) for idx_num in cons_idx]
    cond_chron = [original_idx.index(idx_num) for idx_num in cond_idx]

    # Check cond chronology:
    cons_chron_check = True
    for i in range(1, len(cons_chron)):
        if cons_chron[i-1] - cons_chron[i] != -1:
            cons_chron_check, false_index_cons = False, i
    cond_chron_check = True
    for i in range(1, len(cond_chron)):
        if cond_chron[i-1] - cond_chron[i] != -1:
            cond_chron_check, false_index_cond = False, i
    #print('cond_chron_check: ', cond_chron_check, 'cons_chron_check: ', cons_chron_check)
    return {'condition': cond, 'consequence': cons}


# Failed attempt to make better condition_consequence_extractor
def get_cond_cons_V2(doc):
    possible_conditions = get_possible_conditions(doc)
    condition, dep_tag = get_condition(possible_conditions)

    # If the tags in dep_tags_split originate from root, the subtree of that root without the tag is the consequence
    root = get_root(doc)
    dep_in_root = False
    consequence = []
    for c1 in root.children:
        if c1.dep_ == dep_tag:
            dep_in_root = True
    if dep_in_root and dep_tag != 'mixed':
        before =[]
        after = []
        for child in root.children:
            if child.dep_ not in dep_tag:
                if child.idx < root.idx:
                    # Append everything before root
                    [before.append(i) for i in child.subtree]
                else:
                    # Append everything after root
                    [after.append(i) for i in child.subtree]
        for i in before:
            consequence.append(i)
        consequence.append(root)
        for i in after:
            consequence.append(i)
    else:
        # In this case just return the difference between the condition and the full sentence
        try:
            condition_idx = [w.idx for w in condition[0]]
        except:
            condition_idx = [w.idx for w in condition]
        consequence = [word for word in doc if word.idx not in condition_idx]

    return {'condition': condition, 'consequence': consequence}

# Get the one condition out of get_possible_conditions
def get_condition(possible_conditions):
    if possible_conditions['conj'] != []:
        for key in possible_conditions:
            if possible_conditions[key] != [] and key != 'conj':
                for w in possible_conditions['conj'][0]:
                    possible_conditions[key][0].append(w)
                return possible_conditions[key][0], 'mixed'

    if possible_conditions['advcl'] != []:
        return possible_conditions['advcl'], 'advcl'
    for key in possible_conditions:
        if possible_conditions[key] != []:
            return possible_conditions[key], str(key)

##############################################
# Functions that may be deleted in the future:
def extract_condition_consequence_3(doc):
    condition = []
    consequence = []
    for token in doc:
        if token.dep_ == 'prep':
            [condition.append(child) for child in token.subtree]
        if token.dep_ == 'ROOT':
            before =[]
            after = []
            for child in token.children:
                if child.dep_ != 'prep':
                    if child.idx < token.idx:
                        [before.append(i) for i in child.subtree]
                    else:
                        [after.append(i) for i in child.subtree]
            for i in before:
                consequence.append(i)
            consequence.append(token)
            for i in after:
                consequence.append(i)
    return {'condition': condition, 'consequence': consequence}


def extract_condition_consequence_4(doc):
    """""""""
    --> This function is temporarily broken (23/04)
    This function was made in order to correctly extract this example:
    It's very simple, if the student needs to commute, then the student has right of a permit.
    {'condition': [if, the, student, needs, to, commute], 'consequence': [,, then, the, student, has, right, of, a, permit]}
    """""""""
    for word in doc:
        if word.dep_ == 'advcl' or word.dep_ == 'prep':
            advcl_head = word.head
            consequence = [i for i in advcl_head.subtree]
            condition = [i for i in word.subtree]

            condition_sentence = ''
            consequence_sentence = ''
            for word in condition:
                condition_sentence += word.text + ' '
            for word in consequence:
                consequence_sentence += word.text + ' '

    if condition_sentence in consequence_sentence:
        if condition_sentence.find(consequence_sentence) == 0:
            consequence = consequence[0:len(condition)+1]
        else:
            consequence = consequence[consequence.index(condition[-1])+1:]
    return {'condition': condition, 'consequence': consequence}


#  ██████  ██████  ███    ██ ██████  ██ ████████ ██  ██████  ███    ██
# ██      ██    ██ ████   ██ ██   ██ ██    ██    ██ ██    ██ ████   ██
# ██      ██    ██ ██ ██  ██ ██   ██ ██    ██    ██ ██    ██ ██ ██  ██
# ██      ██    ██ ██  ██ ██ ██   ██ ██    ██    ██ ██    ██ ██  ██ ██
#  ██████  ██████  ██   ████ ██████  ██    ██    ██  ██████  ██   ████

# ███████ ██   ██ ████████ ██████   █████   ██████ ████████  ██████  ██████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██    ██ ██   ██
# █████     ███      ██    ██████  ███████ ██         ██    ██    ██ ██████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██    ██ ██   ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██     ██████  ██   ██

# ██       ██████  ██     ██     ██      ███████ ██    ██ ███████ ██
# ██      ██    ██ ██     ██     ██      ██      ██    ██ ██      ██
# ██      ██    ██ ██  █  ██     ██      █████   ██    ██ █████   ██
# ██      ██    ██ ██ ███ ██     ██      ██       ██  ██  ██      ██
# ███████  ██████   ███ ███      ███████ ███████   ████   ███████ ███████

# Removal of conditional words
def remove_conditional_words(doc):
    """
    Always input a sentence, no list of word tokens
    """
    if 'spacy' in str(type(doc)).lower():
        doc_string = doc.text
    if condition_identifier(doc_string):

        if_then_synonyms_words = ['if', 'whenever', 'wherever', 'when', 'unless', 'presuming']
        if_then_synonyms_phrases = ['in the case that','assuming that', 'conceding that ', 'granted that', 'in case that', 'on the assumption that', 'supposing that ', 'in case of ', 'in the case of ', 'in the case that', 'on condition that ', 'on the condition that', 'given that', 'if and only if ', 'presuming that', 'providing that', 'provided that', 'contingent on ', 'whenever that', 'in the event that']

        # Check words
        tokenized_sentence = [word for word in doc]
        remove_index = []
        for sentence_word in tokenized_sentence:
            if sentence_word.text.lower() in if_then_synonyms_words:
                for i in range(len(tokenized_sentence)):
                    if tokenized_sentence[i].text.lower() in if_then_synonyms_words:
                        remove_index.append(i)
                return ' '.join([w.text for w in remove_elements(tokenized_sentence, remove_index)])
            else:
                for wordphrase in if_then_synonyms_phrases:
                    if wordphrase in doc_string.lower():
                        output = doc_string[:doc_string.lower().find(wordphrase)] + doc_string[doc_string.lower().find(wordphrase) + len(wordphrase):]
                        return output.strip()
        return doc.text

def remove_consequence_words(doc):
    """
    Always input a sentence, no list of word tokens
    """
    if 'spacy' in str(type(doc)).lower():
        doc_string = doc.text
    tokenized_sentence = [word for word in doc]
    tokenized_sentence_strings = [word.text.lower() for word in doc]
    if 'then' in tokenized_sentence_strings:
        # Check words
        remove_index = []
        for i in range(len(tokenized_sentence)):
            if tokenized_sentence[i].text.lower() == 'then':
                remove_index.append(i)
        return ' '.join([w.text for w in remove_elements(tokenized_sentence, remove_index)])
    return doc.text

###################################################################################################
# Functions for extraction of object and its condition
def clean_low_level(condition):
    output = []
    not_in_output = False
    # Check for negations
    for el in condition:
        for i in el:
            if i.text in 'not' or i.text in "n't":
                output.append(i)
                not_in_output = True
    # Clean condition part from if statements
    if_then_synonyms_words = ['if', 'whenever', 'wherever', 'then', 'when', 'unless']
    if_then_synonyms_phrase = ['assuming that ', 'conceding that ', 'granted that ', 'in case that ', 'on the assumption that ', 'supposing that ', 'in case of ', 'in the case of ', 'in the case that ']
    pop_list = []
    for i in range(0, len(condition)):
        for word in condition[i]:
            if word.text.lower() in if_then_synonyms_words or word.text.lower() in if_then_synonyms_phrase:
                pop_list.append(i)
    for i in sorted(pop_list, reverse=True):
        del condition[i]
    if len(condition) >= 2 and not_in_output:
        output.append(condition[1:])
    elif len(condition) >= 2 and not not_in_output:
        output.append(condition[0:])
    elif len(condition) == 1:
        output.append(condition)
    return output


def flatten(nested_list):
    flat_list = []
    for sublist in nested_list:
        if type(sublist) == list:
            for item in sublist:
                flat_list.append(item)
        else:
            flat_list.append(sublist)
    for el in flat_list:
        if type(el) == list:
            return flatten(flat_list)
    return flat_list


def get_ners(spacydoc):
    return [(x.text, x.label_) for x in spacydoc.ents]


def sent_splitter(sent, dep_tags_split):
    parts = []
    for word in sent:
        if word.dep_ == dep_tags_split and word.head.pos_ in ['AUX', 'VERB']:
            parts.append([token for token in word.subtree])
    distinct_parts = []
    # Make list of sent constituent parts
    for part in parts:
        distinct_parts.append(part)
    # Get full and string
    full_and_string = []
    for part in parts:
        [full_and_string.append(word) for word in part]
    distinct_parts.insert(0,get_distinct_sentences(sent, full_and_string)[0])# Get first part sentence
    return distinct_parts
# ==================================================================================================
# Functions to extract binary conditions
def get_lower_level_cond(only_cond):
    only_cond_string = make_string(only_cond)
    # Search for root
    root = get_root(only_cond_string)
    object_or_person = []
    condition = []
    binary_classifier = []
    # Search for binary binary_classifier AND/OR
    for word in only_cond_string:
        if word.pos_ == 'CCONJ' and word.dep_ == 'cc' and word.head.pos_ in ['AUX', 'VERB']:
            binary_classifier.append(word)

    # Search for objects and conditions in the form of {cond1:{'object/person', 'cond', 'binder'}, cond2...}
    distinct_parts = sent_splitter(only_cond, 'conj')
    conds = {'conds':[]}
    i = 1
    for distinct in distinct_parts:
        # Check if there's a verb in the distinct
        conds['conds'].append({f'C{i}': get_object_condition(distinct)})
        i += 1
    if binary_classifier == []:
        binary_classifier = None

    conds['conjs'] = binary_classifier
    #return [conds, {'conjs': binary_classifier}]
    return conds


def get_object_condition(only_cond):
    only_cond = make_string(only_cond)
    # Search for root
    root = get_root(only_cond)
    object_or_person = []
    condition = []
    for sub in root.subtree:
        if sub.dep_ in ['nsubj', 'nsubjpass']:
            object_or_person.append([el for el in sub.subtree])
        elif sub.dep_ in ['acomp', 'attr', 'dobj', 'neg', 'prep', 'advmod']:
            condition.append([el for el in sub.subtree])
    condition = flatten(clean_low_level(condition))
    condition = remove_duplicate_chunks(condition)
    return {'object_or_person': flatten(object_or_person), 'c': condition, 'binder': (root, root.lemma_)}
####################################################################################################
####################################################################################################
# Functions for extraction of object and its consequence
def get_lower_level_cons(only_cons):
    only_cons_string = make_string(only_cons)
    # Search for root
    object_or_person = []
    consequence = []
    binary_classifier = []

    # Search for binary binary_classifier AND/OR
    for word in only_cons_string:
        if word.pos_ == 'CCONJ' and word.dep_ == 'cc' and word.head.pos_ in ['AUX', 'VERB']:
            binary_classifier.append(word)

    # Search for objects and conditions in the form of {cond1:{'object/person', 'cons', 'binder'}, cond2...}
    distinct_parts = sent_splitter(only_cons, 'conj')
    cons = {'cons':[]}
    i = 1
    for distinct in distinct_parts:
        # Check if there's a verb in the distinct
        cons['cons'].append({f'C{i}': get_object_consequence(distinct)})
        i += 1
    if binary_classifier == []:
        binary_classifier = None
    cons['conjs'] = binary_classifier
    #return [cons, {'conjs': binary_classifier}]
    return cons


def get_object_consequence(consequence):
        only_cons = make_string(consequence)
        # Search for root
        root = get_root(only_cons)
        object_or_person = []
        consequence = []
        for sub in root.subtree:
            if sub.dep_ in ['nsubj', 'nsubjpass']:
                object_or_person.append([el for el in sub.subtree])
            elif sub.dep_ in ['xcomp', 'prep', 'attr', 'dobj', 'npadvmod']:
                consequence.append([el for el in sub.subtree])
        consequence = flatten(clean_low_level(consequence))
        consequence = remove_duplicate_chunks(consequence)
        return {'object_or_person': flatten(object_or_person), 'c': consequence, 'binder': (root, root.lemma_)}


def get_rule(low_cond, low_cons):
    # Work on subjects, if 'it', 'he', 'their', 'her', 'them', 'its', 'they',
    objects_persons = []
    if len(low_cond['conds']) == len(low_cons['cons']):
        for key in low_cond['conds']:
            if
    return objects_persons

####################################################################################################

testsentence = sp("If the customer is a business and the order size is less than 10, then the discount is 10%. Whenever the order is less than or equal to 10, discount is 15%. Lastly, the discount amounts to 5% if the customer is private.")

for sent in [sent for sent in testsentence.sents]:
    print(condition_consequence_extractor_v4(sent))



# ██████  ██ ██████  ███████ ██      ██ ███    ██ ███████
# ██   ██ ██ ██   ██ ██      ██      ██ ████   ██ ██
# ██████  ██ ██████  █████   ██      ██ ██ ██  ██ █████
# ██      ██ ██      ██      ██      ██ ██  ██ ██ ██
# ██      ██ ██      ███████ ███████ ██ ██   ████ ███████


# --------------------------------------------------------------------------------------------------
# # Data processing part
# %% Extract all data from the text data and make list ---------------------------------------------
#filename = 'C:/Users/Arnaud/Google Drive/Master of AI/3. Thesis/thesis_2/raw_data.txt'


temp_list = get_texts('textual_data/raw_data.txt')
# %% Make data dict --------------------------------------------------------------------------------
# Remove unnecesarry characters

only_sentences = get_only_sentences(temp_list)
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

sentences_spacy = get_spacy_lib(only_sentences)
# --------------------------------------------------------------------------------------------------
# Analysis of above dicts -----------------------------------------------------------------------
# list(sentences_nltk['Dataset_1'])
# list(sentences_STNLP['Dataset_1']) # Seems best (clear Segmentation)
# list(sentences_spacy['Dataset_1'])

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
# print(list(sentences_tokenized_nltk['Dataset_1'])[:5])
# print(list(sentences_tokenized_stNLP['Dataset_1'])[:5])
# print(list(sentences_tokenized_spacy['Dataset_1'])[:5])
# print(list(sentences_tokenized_corenlp['Dataset_1'])[:5]) # Doesn't make separation with "-"
#     # --> There doesn't seem to be a difference, spacy list is a spacy object btw

# --------------------------------------------------------------------------------------------------
# Removing punctuations ----------------------------------------------------------------------------
sentences_tokenized_nltk_clean = remove_puncts(sentences_tokenized_nltk)
sentences_tokenized_stNLP_clean = remove_puncts(sentences_tokenized_stNLP)
sentences_tokenized_spacy_clean = remove_puncts(sentences_tokenized_spacy)
sentences_tokenized_corenlp_clean = remove_puncts(sentences_tokenized_corenlp)
# --------------------------------------------------------------------------------------------------
# %% POS tagging  ----------------------------------------------------------------------------------

# ....... with NLTK (only here distinction possible between clean and unclean tokenized sentences)

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
sentence_normal = 'In case that a person is between 19 and 21 years old and was not involved in a car accident, car insurance costs 500 euros.'
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


# %% ....... with CoreNLP ----> Gives full parse (dependency)
sentence_normal = "It is very simple, if the student needs to commute, then the student has right of a permit."
nlp_wrapper = StanfordCoreNLP(r'../thesis/stanfordfiles/stanford-corenlp-full-2017-06-09')
corenlp_depparse = nlp_wrapper.annotate(sentence_normal, properties={'annotators': 'depparse', 'outputFormat': 'json'})
nlp_wrapper.close()

workdoc = json.loads(corenlp_depparse)['sentences']


# %%....... with spacy -----------------------------------------------------------------------------
#texts = [only_sentences[3]]
#text = texts[0]
#doc = sp(texts)
doc = sp("A car driver needs to pay a fine of 20 euro if he had an accident.")

dep_df = get_dep_df(doc)

# %% Before continuing, I need other information of the individual sentences since in one senctence, there could be references to the same entity
# neuralcoref.add_to_pipe(sp)
# corefs = doc._.coref_clusters

# Extraction of seperate elements and their meta information
# Now begin extracting the actual objects and what their requirements argument
# For this, I'll need some kind of semantic parsing since it has to do with meaning

# Clean/prepare extracted parts
###################################################################################################
# %% Conditional statement handler ################################################################
# First look at what the sentence handles about -> A person(s) or item(s)?
"""""""""
Does it concern multiple items or persons?
Are there AND/OR statements
"""""""""

# --------------------------------------------------------------------------------------------------
# Ex 1: If A then ACTION --> So only one object, one condition for object and one action
doc = sp("A boat needs to be checked if it hasn't an age of 20 years.")
cond_cons = condition_consequence_extractor(doc)
get_lower_level_cond(cond_cons['condition'])
get_lower_level_cons(cond_cons['consequence'])

# --------------------------------------------------------------------------------------------------
# Ex 2: If A and B then ACTION / If A or B then ACTION (SAME DEPENDENCY STRUCTURE FOR AND AND OR)

# I need to be able to search for synonyms of "no", "and" and "or"
doc = sp("If the person is between 22 and 29 years of age, and was involved in a car accident, insurance cost is 600 euros.")
cond_cons = condition_consequence_extractor(doc)
get_lower_level_cond(cond_cons['condition'])


# --------------------------------------------------------------------------------------------------
# %% consequence staments handler ------------------------------------------------------------------
# Then look at what condition the item/person should be in
doc = sp("In case that a person is between 19 and 21 years old and was not involved in a car accident, car insurance costs 500 euros.")
cond_cons = condition_consequence_extractor(doc)

only_cons = make_string(cond_cons['consequence'])
get_dep_parse(only_cons)
get_lower_level_cons(cond_cons['consequence'])

###################################################################################################
###################################################################################################

# All key elements in place, now last representation of the rules
doc = sp("If the service request is a product change and the customer is in , the customer is charged 50 euro.")
cond_cons = condition_consequence_extractor(doc)

low_cond = get_lower_level_cond(cond_cons['condition'])
low_cons = get_lower_level_cons(cond_cons['consequence'])

get_rule(low_cond, low_cons)


###################################################################################################
###################################################################################################

temp_set = ['The car needs to be washed if it is blue.', 'The student needs to pay 30 euro if he is 21 years old.', 'If he had an accident, the car driver needs to pay 30 euro.']

####################################################################################################
# PRELIMINARY RESEARCH PART 1 ######################################################################
####################################################################################################
if_then_synonyms_words = ['if', 'whenever', 'wherever', 'then', 'when', 'unless']
if_then_synonyms_phrase = ['in the case that ','assuming that ', 'conceding that ', 'granted that ', 'in case that ', 'on the assumption that ', 'supposing that ', 'in case of ', 'in the case of ', 'in the case that ', 'on condition that ', 'on the condition that ', 'given that ', 'if and only if ', 'presuming that ', 'presuming ', 'providing that ', 'provided that ', 'contingent on ', 'whenever that ', 'in the event that ']

testsent_1 = "In case that the upcoming days are sunny, salad should be bought."
testsent_2 = "In the case that the upcoming days are sunny, salad should be bought."
testsent_3 = "Assuming that the upcoming days are sunny, salad should be bought."
testsent_4 = "Whenever the upcoming days are sunny, salad should be bought."
testsent_5 = "Conceding that the upcoming days are sunny, salad should be bought."
testsent_6 = "Given that the upcoming days are sunny, salad should be bought."
testsent_7 = "Provided that the upcoming days are sunny, salad should be bought."
testsent_8 = "Granted that the upcoming days are sunny, salad should be bought."

test_sent_root = "If the upcoming days are sunny, salad should be bought."

testsents = [testsent_1, testsent_2, testsent_3, testsent_4, testsent_5, testsent_6, testsent_7, testsent_8]

####################################################################################################
# PRELIMINARY RESEARCH PART 2 ######################################################################
####################################################################################################

core_sent = "Salad should be bought if the upcoming days are sunny."
core_sent = "A different supplier should be chosen if he does not comply with our requirements."

testsent_1 = "A different supplier should be chosen, in case that he does not comply with our requirements and committed a crime."
testsent_2 = "A different supplier should be chosen, in the case that he does not comply with our requirements and committed a crime."
testsent_3 = "A different supplier should be chosen, assuming that he does not comply with our requirements and committed a crime."
testsent_4 = "A different supplier should be chosen, whenever he does not comply with our requirements and committed a crime."
testsent_5 = "A different supplier should be chosen, conceding that he does not comply with our requirements or committed a crime."
testsent_6 = "A different supplier should be chosen, given that he does not comply with our requirements or committed a crime."
testsent_7 = "A different supplier should be chosen, provided that he does not comply with our requirements or committed a crime."
testsent_8 = "A different supplier should be chosen, granted that he does not comply with our requirements or committed a crime."
testsent_9 = "Salad should be bought, in case that the upcoming days are sunny and the supplier is reliable."
testsent_10 = "Salad should be bought, in the case that the upcoming days are sunny and the supplier is reliable."
testsent_11 = "Salad should be bought, assuming that the upcoming days are sunny and the supplier is reliable."
testsent_12 = "Salad should be bought, whenever the upcoming days are sunny and the supplier is reliable."
testsent_13 = "Salad should be bought, conceding that the upcoming days are sunny or the supplier is reliable."
testsent_14= "Salad should be bought, given that the upcoming days are sunny or the supplier is reliable."
testsent_15 = "Salad should be bought, provided that the upcoming days are sunny or the supplier is reliable."
testsent_16 = "Salad should be bought, granted that the upcoming days are sunny or the supplier is reliable."

tests = "Unless the person is sick, he should take public transport."

testlist = [testsent_1, testsent_2, testsent_3, testsent_4, testsent_5, testsent_6, testsent_7, testsent_8, testsent_9, testsent_10, testsent_11, testsent_12, testsent_13, testsent_14, testsent_15, testsent_16]

########################################################################################################################################################################################################

# Implied logical sentences
test1 = sp("A new employee receives a mobile phone.")
test2 = sp("5 euros are always charged.")
test3 = sp("Costs should always be compensated.")
test4 = sp("Laptops are repared for free.")
test5 = sp("Employees that achieved their goals always are rewarded.")
test6 = sp("New customers should always receive a discount.")
test7 = sp("A sick employee should always stay at home.")
test8 = sp("Orders above 10 euro should be shipped for free.")
test9 = sp("Orders below 10 euro have a shiping cost of 2 euro.")
test10 = sp("Merchandise that was bought 4 months ago should be cleaned.")

implieds = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10]

####################################################################################################
testsss = sp("An employee should receive a bonus given that he achieved his yearly sales quota.")

####################################################################################################

exsent_1 = sp("The order is not more than 30 units.")
exsent_2 = sp("The order doesn't exceed more than 30 units.")
exsent_3 = sp("The order isn't exceeding more than 30 units.")
exsent_4 = sp("The order exceeds not more than 30 units.")
exsent_5 = sp("The order won't exceed than 30 units.")
exsent_6 = sp("The service request is not a bug.")
exsent_7 = sp("The service request isn't a bug.")
exsent_8 = sp("The service request does not constitute a bug.")
exsent_9 = sp("The service request can't be seen as a bug.")
exsent_10 = sp("The service request can not be a bug.")

not_sentences = [exsent_1, exsent_2, exsent_3, exsent_4, exsent_5, exsent_6, exsent_7, exsent_8, exsent_9, exsent_10]
####################################################################################################
# DEVELOPMENT OF LOWER LEVEL EXTRACTOR

# Simple case ######################################################################################

# If X then Y
sl_1 = "Whenever the order amount is 10, a discount of 2% will apply."
# --> order amount = 10 --> discount = 2%
sl_2 = "If the order amount is equal to 10, then a discount of 2% will be given."
# --> order amount = 10 --> discount = 2%
sl_3 = "Granted that the order amount is more than 10, a discount of 6% will apply."
# --> order amount > 10 --> discount = 6%
sl_4 = "No discount will apply in case that the order amount is less than 10."
# --> order amount < 10 --> discount = None
sl_5 = "If the order amount is between 10 and 20, the discount will be 7%."
# --> order amount = [10, 20] --> discount = 7%
sl_6 = "If the order amount is less than or equal to 40, a discount of 8% will apply."
# --> order amount <= 40 --> discount = 8%
sl_7 = "If the amount of the order is more than or equal to 30, a discount of 4% will be applied."
# --> order amount >= 30 --> discount = 4%
sl_8 = "If the customer ordered more than or equal to 30, a discount of 4% will be applied."
# --> order amount >= 30 --> discount = 4%
sl_9 = "If the customer's order is greater than or equal to 30, a discount of 4% will be applied."
# --> order amount >= 30 --> discount = 4%

sl_list = [sl_1, sl_2, sl_3, sl_4, sl_5, sl_6, sl_7, sl_8, sl_9]

####################################################################################################

validation = ["If the mouse is red then it should be washed.", "Whenever the day is Sunday, the discount is 10%.", "In case that the laptop is green, the ball is red.", "When it is weekend, the shop does not open.", "If the temperature is between 5 and 10 degrees, then the ice should be refrigerated."]

####################################################################################################
# sentences_spacy['Dataset_1']
# Test:

sentences_spacy['Dataset_8']
# A discount of 4% and otherwise 9% is given when the order exceeds 10 units.
for sentence in ["Whenever the customer is in the testing phase, and the customer's service request is a change request, the cost is 600 euros."]:
    print('----- NEXT SENTENCE -----')
    print(sentence)
    if 'spacy' not in str(type(sentence)).lower():
        sentence = sp(sentence)
    cond_cons = condition_consequence_extractor_v4(sentence)
    cond = sp(remove_conditional_words(make_string(cond_cons['condition'])))
    cons = sp(remove_consequence_words(make_string(cond_cons['consequence'])))
    #get_dep_parse(cond)
    #get_dep_parse(cons)
    print('--cond--')
    print(cond)
    print('--cons--')
    print(cons)
    print('RULE:')
    print(get_full_dmn_rule(sentence))



def split_in_conjs(doc):
    doc_dep_tags = [w.dep_ for w in doc]
    doc_pos_tags = [w.pos_ for w in doc]
    # Conjunctions? or Disjunctions?
    # -- If there is no conjunction tag, return the original doc
    if 'conj' not in doc_dep_tags:
        return doc
    else:
        # Split in possible conjunctions
        while 'conj' in 
        conj = [get_subtree(doc, 'conj')]
        return 'DONE'


def get_full_dmn_rule(doc):
    rule = {'if':[], 'then':[], 'else':[]}
    conds, conss, elses, else_part = [], [], [], []

    #  Step 1: get the cond, cons and else parts
    cond_cons = condition_consequence_extractor_v4(doc)
    cond, cons = sp(remove_conditional_words(make_string(cond_cons['condition']))), sp(remove_consequence_words(make_string(cond_cons['consequence'])))
    else_in_cons, else_syn = else_identifier(str(cons))
    if else_in_cons:
        cons, else_part = extract_else_phrase(cons, else_syn)

    # Step 2: split up the conds and cons in its elements (conjunction/disjunction)
    # -- Step 2.a for the cond
    print('--------------------------')
    print(split_in_conjs(cond))
    conds.append(cond)

    # -- Step 2.b for the cons
    print(split_in_conjs(cons))
    conss.append(cons)

    # -- Step 2.c for the else
    print(split_in_conjs(else_part))
    elses.append(else_part)
    print('--------------------------')

    # Step 3: Extract and append the rule notations to the appropriate key in dict
    # -- 3.a ifs
    for c in conds:
        rule['if'].append(get_lower_level_rule(c))
    # -- 3.b thens
    for c in conss:
        rule['then'].append(get_lower_level_rule(c))
    # -- 3.c elses
    for c in elses:
        rule['else'].append(get_lower_level_rule(c))
    return rule


def extract_else_phrase(cons, else_syn):
    if len([sent for sent in cons.sents]) == 1:
        principal_part = [word for word in cons[:[w.text.lower() for w in cons].index(else_syn)]]
        else_part = get_other_part(cons, principal_part)
    else:
        principal_part = [sent for sent in cons.sents][0]
        else_part = [sent for sent in cons.sents][1]
    return make_string(principal_part), make_string(else_part)


def num_or_nom(doc):
    pos_tags = [w.pos_ for w in doc]
    for pos in pos_tags:
        if pos == 'NUM':
            return 'NUM'
    return 'NOM'


def get_lower_level_rule(doc):
    if doc != []:
        if num_or_nom(doc) == 'NUM':
            return get_dmn_rule_num(doc)
        elif num_or_nom(doc) == 'NOM':
            return get_dmn_rule_nom(doc)
    else:
        return doc


# No conjunctions --> If A = a then B = b
def get_dmn_rule_num(doc):
    doc_idx = [w.idx for w in doc]
    possible_vars = []
    possible_vals = []
    possible_verbs = []
    rule_sign = get_rule_sign(doc)
    root_word = get_root(doc)
    doc_dep_tags = [w.dep_ for w in doc]
    #print('---')
    # If there's only one element in the doc, then it's probably the value itself
    #if 'nsubj' not in doc_dep_tags and 'nsubjpass' not in doc_dep_tags:
    #    possible_vals.append([w for w in doc if w.pos_ in ['NOUN', 'ADJ', 'NUM']])
    #    return possible_vars, rule_sign, possible_vals

    # Parse over the sentence and append every word to the right list
    if root_word.pos_ == 'NOUN':
        possible_vars.append(root_word)
        for w in doc:
            if w.dep_ in ['attr', 'pobj', 'acomp', 'dobj', 'nsubj']:
                    possible_vals.append([subw for subw in w.subtree if subw.pos_ in ['NOUN', 'ADJ', 'NUM', 'PROPN']])
    elif 'dobj' not in doc_dep_tags and 'pobj' not in doc_dep_tags and 'attr' not in doc_dep_tags:
        for w in doc:
            if w.dep_ == 'nsubj' or w.dep_ == 'nsubjpass':
                possible_vals.append([wi for wi in w.subtree])
    else:
        for w in doc:
            if w.pos_ in ['VERB', 'AUX']:
                possible_verbs.append({'word': w, 'lemma': sp(w.lemma_), 'id': w.idx, 'index': doc_idx.index(w.idx)})
                if sp(w.lemma_)[0].pos_ == 'NOUN':
                    possible_vars.append([sp(w.lemma_)])
            # Append possible vars
            elif w.dep_ in ['nsubj', 'nsubjpass']:
                possible_vars.append([subw for subw in w.subtree if subw.pos_ in ['NOUN', 'ADJ', 'PRON']])
            # Append possible vals
            elif w.dep_ in ['attr', 'pobj', 'acomp', 'dobj']:
                possible_vals.append([subw for subw in w.subtree if subw.pos_ in ['NOUN', 'ADJ', 'NUM', 'PROPN']])
    #flatten all intermediate outputs before cleaning them
    possible_vars = flatten(possible_vars)
    possible_vals = flatten(possible_vals)
    #print('vars: ', possible_vars)
    #print('vals: ', possible_vals)
    if possible_vals == []:
        possible_vars.append([possible_verbs_i['lemma'] for possible_verbs_i in possible_verbs])
        possible_vals = get_true_or_false(doc)
        possible_vars = flatten(possible_vars)[-1]
    elif len(possible_vals) > 1:
        possible_vals = clean_vals(possible_vals)
    possible_vars = flatten(possible_vars)

    return possible_vars, rule_sign, possible_vals


def get_dmn_rule_nom(doc):
    doc_idx = [w.idx for w in doc]
    possible_vars = []
    possible_vals = []
    possible_verbs = []
    rule_sign = get_rule_sign(doc)
    root_word = get_root(doc)
    doc_dep_tags = [w.dep_ for w in doc]
    #print('---')
    # If there's only one element in the doc, then it's probably the value itself
    #print(doc_dep_tags)
    if 'nsubj' not in doc_dep_tags and 'nsubjpass' not in doc_dep_tags:
        possible_vals.append([w for w in doc if w.pos_ in ['NOUN', 'ADJ', 'NUM']])
        return possible_vars, rule_sign, possible_vals

    # Parse over the sentence and append every word to the right list
    for w in doc:
        if w.pos_ in ['VERB', 'AUX']:
            possible_verbs.append({'word': w, 'lemma': sp(w.lemma_), 'id': w.idx, 'index': doc_idx.index(w.idx)})
            if sp(w.lemma_)[0].pos_ == 'NOUN':
                possible_vars.append([sp(w.lemma_)])
        # Append possible vars
        elif w.dep_ in ['nsubj', 'nsubjpass']:
            possible_vars.append([subw for subw in w.subtree if subw.pos_ in ['NOUN', 'ADJ', 'PRON']])
        # Append possible vals
        elif w.dep_ in ['attr', 'pobj', 'acomp', 'dobj']:
            possible_vals.append([subw for subw in w.subtree if subw.pos_ in ['NOUN', 'ADJ', 'NUM', 'PROPN']])
    #flatten all intermediate outputs before cleaning them
    possible_vars = flatten(possible_vars)
    possible_vals = flatten(possible_vals)
    #print('vars: ', possible_vars)
    #print('vals: ', possible_vals)
    if possible_vals == []:
        possible_vars.append([possible_verbs_i['lemma'] for possible_verbs_i in possible_verbs])
        possible_vals = get_true_or_false(doc)
        possible_vars = flatten(possible_vars)[-1]
    elif len(possible_vals) > 1:
        possible_vals = clean_vals(possible_vals)
    possible_vars = flatten(possible_vars)

    return possible_vars, rule_sign, possible_vals
# No conjunctions --> If A = a then B = b else B = c


def get_true_or_false(doc):
    for w in doc:
        if w.dep_ == 'neg':
            return False
        elif 'no' in [w.text.lower() for w in doc]:
            return False
    return True


def clean_vals(possible_vals):
    possible_vals_idx = [w.idx for w in possible_vals]
    for i in possible_vals_idx:
        if possible_vals_idx.count(i) > 1:
            return possible_vals[possible_vals_idx.index(i)]
    possible_vals_str = ' '.join([w.text for w in possible_vals])
    for sent in ['great than equal', 'more than equal', 'less than equal', 'great than', 'more than', 'less than']:
        if sent in possible_vals_str:
            return possible_vals_str[possible_vals_str.find() + len(sent):]
    return possible_vals


def get_rule_sign(doc):
    doc_text = ' '.join([w.lemma_ for w in doc]).lower()
    less_than_syns = ['less than', 'few than', "do n't exceed ", "do 'nt surmount ", "don't pass "]
    less_equal_syns = ['less than or equal', 'less than or equal']
    greater_than_syns = ['great than', 'more than', 'exceed ', 'surmount ', ' pass ', 'be above']
    great_equal_syns = ['great than or equal', 'more than or equal']
    if 'between ' in doc_text or 'within ' in doc_text or 'interval ' in doc_text:
        return '[X, Y]'
    for great_equal in great_equal_syns:
        if great_equal in doc_text:
            return '>='
    for great_syn in greater_than_syns:
        if great_syn in doc_text:
            return '>'
    for less_equal in less_equal_syns:
        if less_equal in doc_text:
            return '=<'
    for less_syn in less_than_syns:
        if less_syn in doc_text:
            return '<'
    return '='
