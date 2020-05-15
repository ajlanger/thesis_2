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
import spacy
from spacy import displacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------------------------------
# %% Special settings & miscellanneous -------------------------------------------------------------
spacy.prefer_gpu()
#nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
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


def get_biggest_subtree(doc, tag):
    output = []
    for word in doc:
        if word.dep_ == tag:
            output.append([w for w in word.subtree])
    output_len = [len(el) for el in output]
    output = output[output_len.index(max(output_len))]
    return output


def remove_puncts_v2(doc_list):
    output = [w for w in doc_list if w.pos_ != 'PUNCT']
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

# %% HIGH LEVEL
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


def condition_consequence_extractor_v4(doc):
    if condition_identifier(doc):
        doc_dep_tags = [word.dep_ for word in doc]
        possible_conditions, possible_ors = get_possible_conditions(doc)
        # If the root indicates the consequence
        if not dict_empty(possible_conditions) and 'relcl' not in doc_dep_tags:
            condition_part, split_keys = get_condition_v4(doc, possible_conditions, possible_ors)
            #consequence_part = get_root_subtree_without_tagx(doc, split_keys)
            consequence_part = get_other_part(doc, condition_part)
        elif 'relcl' in doc_dep_tags:
            # If root indicates the condition
            consequence_part = get_biggest_subtree(doc, 'relcl')
            condition_part = get_other_part(doc, consequence_part)
        else:
            return 'No conditional statements could be extracted in spite of a condition being present.'
        ############################################################################################
        # Probably not needed anymore
        # Perform cleaning on output:
        output = {'condition': remove_duplicate_chunks(condition_part), 'consequence': remove_duplicate_chunks(consequence_part)}
        disctinct_sentences = get_distinct_sentences_v2(output['condition'], output['consequence'])
        output = {'condition': remove_link_words(disctinct_sentences[0]), 'consequence': remove_link_words(disctinct_sentences[1])}

        return output
    else:
        return 'No conditional statement in sentence'
        #return implied_condition_consequence_extractor(doc)

def get_other_part(doc, first_part):
    doc_idx = [word.idx for word in doc]
    first_part_idx = [word.idx for word in first_part]
    output = []
    for word in doc:
        if word.idx not in first_part_idx:
            output.append(word)
    return output


def extract_correct_advcl_part(doc):
    temp_list_output = []
    for word in doc:
        if word.dep_ in ['advcl']:
            temp_list = []
            before = []
            after = []
            for w in [child for child in word.children]:
                if w.dep_ != 'advcl':
                    if w.idx < word.idx:
                        # Append everything before root
                        [before.append(wi) for wi in w.subtree]
                    else:
                        # Append everything after root
                        [after.append(wi) for wi in w.subtree]
            temp_list.append([w for w in before])
            temp_list.append(word)
            temp_list.append([w for w in after])
            if condition_identifier(make_string(flatten(temp_list))):
                temp_list_output.append(flatten(temp_list))
    return temp_list_output


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
    #elif 'prep' in keys:
        #key = 'prep'
        if len(possible_conditions[key]) > 1:
            possible_conditions[key] = extract_correct_advcl_part(doc)

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
    else_synonyms_phrases = ['any other way ', 'if not ', 'in different circumstances ', 'or else ', 'or then ']

    # Check words
    for sentence_word in nltk.word_tokenize(sentence):
        if sentence_word.lower() in else_synonyms_words:
            return True, sentence_word
        else:
            for wordphrase in else_synonyms_phrases:
                if wordphrase in sentence.lower():
                    return True, wordphrase
    return False, None

# %% Low level
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

def valid_phrase(distinct_conj):
    pos_tags = [el.pos_ for el in distinct_conj]
    dep_tags = [el.dep_ for el in distinct_conj]
    if ('NOUN' in pos_tags and 'VERB' in pos_tags) or ('AUX' in pos_tags and 'NOUN' in pos_tags) or ('NUM' in pos_tags and 'VERB' in pos_tags) or ('NUM' in pos_tags and 'AUX' in pos_tags):
        return True
    elif ('nsubj' in dep_tags or 'nsubjpass' in dep_tags) or ('quantmod' in dep_tags and 'NOUN' in dep_tags):
        return True
    return False


def split_in_conjs(doc):
    binder = None
    if doc == []:
        return [doc], binder
    doc_dep_tags = [w.dep_ for w in doc]
    doc_pos_tags = [w.pos_ for w in doc]
    token_list = [w for w in doc]
    output_parts = []
    conj_words = [w for w in doc if w.pos_ == 'CCONJ']
    # -- If there is no conjunction tag, return the original doc
    if 'conj' not in doc_dep_tags:
        return [doc], binder
    else:
        # Split in possible conjunctions
        while 'conj' in doc_dep_tags:
            conj = get_biggest_subtree(doc, 'conj')
            distinct_conj = make_string(conj)
            last_conj_valid = True
            # Check whether the extracted conj is actually valid new phrase
            if valid_phrase(distinct_conj):
                first_part = remove_puncts_v2(get_other_part(doc, conj))
                distinct_first_part = make_string(first_part)
                output_parts.append(make_string(distinct_first_part))
            else:
                last_conj_valid = False
                # Append the sentence until that chunk to the output and get the difference of that one with the original
                #first_part = doc[:token_list.index(conj[-1])+1]
                first_part = doc[:token_list.index(conj_words[-1])+1]
                conj = remove_puncts_v2(get_other_part(doc, first_part))
                first_part = make_string([w for w in first_part])
                output_parts.append(first_part)
                if conj == []:
                    break
                else:
                    distinct_conj = make_string(conj)
            doc = distinct_conj
            token_list = [w for w in doc]
            doc_dep_tags = [w.dep_ for w in distinct_conj]
            last_conj_valid = True
        if conj == [] and len(output_parts) == 1:
            return [output_parts], None
        else:
            if last_conj_valid:
                output_parts.append(distinct_conj)
            # Search binder
            for l in range(len(output_parts)):
                for i in [-1,0]:
                    if output_parts[l][i].pos_ == 'CCONJ':
                        binder = output_parts[l][i]
                        if i == -1:
                            output_parts[l] = make_string([w for w in output_parts[l][:-1]])
                        elif i == 0:
                            output_parts[l] = make_string([w for w in output_parts[l][0:]])
    return output_parts, binder


def remove_link_words(doc):
    link_words = ['furthermore', 'additionaly', 'besides', 'moreover', 'firstly', 'secondly', 'thirdly', 'fourthly', 'fifthly', 'sixthly', 'further', 'as well as', 'not to mention', 'on the other hand', 'at last', 'finally']
    link_phrases = []
    list_words = [w for w in doc]
    string_words = ' '.join([w.text for w in doc]).lower()
    output = ''
    for phrase in link_words:
        if phrase in string_words:
            output = string_words[:string_words.find(phrase)] + string_words[string_words.find(phrase)+len(phrase):]
            output = output.strip()
            output = sp(output)
            return output
    return doc


def get_full_dmn_rule(doc):
    rule = {'if':[], 'then':[], 'else':[]}
    else_part = []
    #  Step 1: get the cond, cons and else parts
    cond_cons = condition_consequence_extractor_v4(doc)
    if cond_cons == 'No conditional statement in sentence':
        return get_lower_level_rule_v2(doc)
    cond, cons = sp(remove_conditional_words(make_string(cond_cons['condition']))), sp(remove_consequence_words(make_string(cond_cons['consequence'])))
    cond, cons = remove_link_words(cond), remove_link_words(cons)
    else_in_cons, else_syn = else_identifier(str(cons))
    if else_in_cons:
        cons, else_part = extract_else_phrase(cons, else_syn)

    # Step 2: split up the conds and cons in its elements (conjunction/disjunction)
    # -- Step 2.a for the cond
    conds, binder_if = split_in_conjs(cond)
    if binder_if != None:
        rule['binder_if'] = binder_if
    # -- Step 2.b for the cons
    conss, binder_then = split_in_conjs(cons)
    if binder_then != None:
        rule['binder_then'] = binder_then
    # -- Step 2.c for the else
    elses, binder_else = split_in_conjs(else_part)
    if binder_else != None:
        rule['binder_else'] = binder_else
    # Step 3: Extract and append the rule notations to the appropriate key in dict
    # -- 3.a ifs
    for c in conds:
        rule['if'].append(get_lower_level_rule_v2(c))
    # -- 3.b thens
    for c in conss:
        rule['then'].append(get_lower_level_rule_v2(c))
    # -- 3.c elses
    for c in elses:
        rule['else'].append(get_lower_level_rule_v2(c))
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
    less_than_syns = ['less than', 'few than', "do n't exceed ", "do 'nt surmount ", "don't pass ", 'young than', 'low than']
    less_equal_syns = ['less than or equal', 'less than or equal']
    greater_than_syns = ['great than', 'more than', 'exceed ', 'surmount ', ' pass ', 'be above', 'old than']
    great_equal_syns = ['great than or equal', 'more than or equal']
    if 'between ' in doc_text or 'within ' in doc_text or 'interval ' in doc_text:
        return 'interval'
    if 'neg' in [w.dep_ for w in doc]:
        return '!='
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


def get_lower_level_rule_v2(doc):
    if doc == []:
        return []
    get_dep_parse(doc)
    rule_sign = get_rule_sign(doc)
    vars = []
    vals = []
    if rule_sign == 'interval':
        for word in doc:
            if word.pos_ in ['NUM']:
                vals.append(word)
            elif word.pos_ in ['NOUN', 'ADJ']:
                vars.append(word)
    else:
        for word in doc:
            if word.dep_ in ['nsubjpass','nsubj']:
                vars.append([w for w in word.subtree if w.pos_ in ['NOUN', 'ADJ']])
            elif word.dep_ in ['dobj', 'pobj', 'attr', 'acomp']:
                vals.append([w for w in word.subtree if w.pos_ in ['NOUN', 'ADJ', 'NUM', 'PROPN', 'VERB']])
            # For action sentences
            elif word.dep_ in ['xcomp']:
                vars.append([w for w in word.subtree if w.pos_ in ['VERB']])
    vars = flatten(vars)
    vals = flatten(vals)
    if 'NUM' in [w.pos_ for w in doc] and 'NUM' not in [w.pos_ for w in vars] and 'NUM' not in [w.pos_ for w in vals]:
        for w in doc:
            if w.pos_ in ['NUM']:
                vars.append(w)
    if vars == [] and vals == []:
        vars = [w for w in doc if w.pos_ in ['ADJ', 'NOUN', 'NUM']]

    if sp(get_root(doc).lemma_)[0].pos_ == 'NOUN' and (vals == [] or vars == []):
        vars.append(get_root(doc))
    if 'NUM' in [el.pos_ for el in flatten(vars)] and flatten(vals) != []:
        vars, vals = vals, vars
    if flatten(vals) == []:
        rule_sign = '='
        vals = get_true_or_false(doc)
    elif flatten(vars) == []:
        rule_sign = '='
        vars = vals
        vals = get_true_or_false(doc)
    if type(vals) == list and vals != []:
        vals = remove_duplicate_chunks(flatten([vals]))
    if type(vars) == list and vars != []:
        vars = remove_duplicate_chunks(flatten([vars]))
    return vars, rule_sign, vals


####################################################################################################

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
# ....... with spaCy

sentences_spacy = get_spacy_lib(only_sentences)

# A discount of 4% and otherwise 9% is given when the order exceeds 10 units.
for sentence in sentences_spacy['Dataset_8']:
    print('######################################################')
    #print('')
    print('----- NEXT SENTENCE -----')
    #print('')
    print(sentence)
    if 'doc' not in str(type(sentence)).lower():
        sentence = sp(str(sentence))
    cond_cons = condition_consequence_extractor_v4(sentence)

    if cond_cons != 'No conditional statement in sentence' and cond_cons != 'No conditional statements could be extracted in spite of a condition being present.':
        #print('HIGH LEVEL')
        #print(cond_cons)
        cond = sp(remove_conditional_words(make_string(cond_cons['condition'])))
        cons = sp(remove_consequence_words(make_string(cond_cons['consequence'])))
        #print('--cond--')
        #print(cond)
        #print('--cons--')
        #print(cons)
        #print('')
        #print('LOW LEVEL')
        dictiona = get_full_dmn_rule(sentence)
        #print(dictiona)
        print('IF: ', dictiona['if'])
        print('THEN: ', dictiona['then'])
        print('ELSE: ', dictiona['else'])
        # print(dictiona['binder_if'], dictiona['binder_then'], dictiona['binder_else'])
    else:
        print(cond_cons)
        #print(get_lower_level_rule(sentence))
        print('')

input_sentence = sp("If the employee has at least 15 but less than 30 years of service, 2 extra days are given.")
get_dep_parse(input_sentence)
get_possible_conditions(input_sentence)
condition_consequence_extractor_v4(input_sentence)
get_full_dmn_rule(input_sentence)
