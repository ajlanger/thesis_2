# --------------------------------------------------------------------------------------------------
# %% import libraries ------------------------------------------------------------------------------
import json
import nltk
import io
import spacy
from spacy import displacy
import pandas as pd

# --------------------------------------------------------------------------------------------------
# %% Special settings & miscellanneous -------------------------------------------------------------
# Load the spaCy model
spacy.prefer_gpu()
sp = spacy.load('en_core_web_sm')

# --------------------------------------------------------------------------------------------------
# Functions ----------------------------------------------------------------------------------------
#####################################################################################
#####################################################################################
# %% SUPPORT

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
#####################################################################################
#####################################################################################

def get_texts(filename):
    filename = filename
    temp_file = open(filename, 'r').read()
    temp_list = temp_file.split('\n')
    return temp_list


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


def get_spacy_lib(only_sentences):
    sentences_spacy = {}
    for el in range(0, len(only_sentences), 2):
        temp_list = []
        for sentence in sp(only_sentences[el+1]).sents:
            temp_list.append(sentence)
        sentences_spacy[only_sentences[el]] = temp_list
    return sentences_spacy


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
    if type(nested_list) != list:
        return nested_list
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


#####################################################################################
#####################################################################################
# %% HIGH LEVEL

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
#####################################################################################
#####################################################################################

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


def condition_consequence_extractor(doc):
    try:
        if condition_identifier(doc):
            possible_conditions, possible_ors = get_possible_conditions(doc)
            condition_part, split_key = get_condition(possible_conditions)
            consequence_part = get_consequence(doc, split_key)
            # Perform cleaning on output:
            output = {'condition': remove_duplicate_chunks(condition_part), 'consequence': remove_duplicate_chunks(consequence_part)}
            disctinct_sentences = get_distinct_sentences(condition_part, consequence_part)
            output = {'condition': disctinct_sentences[0], 'consequence': disctinct_sentences[1]}

            return output
        else:
            return 'No conditional statement in sentence'
    except:
        return 'No conditional statements could be extracted in spite of a condition being present.'


def get_other_part(doc, first_part):
    doc_idx = [word.idx for word in doc]
    first_part_idx = [word.idx for word in first_part]
    output = []
    for word in doc:
        if word.idx not in first_part_idx:
            output.append(word)
    return output


def get_condition(possible_conditions):
    for key in possible_conditions:
        if possible_conditions[key] != []:
            condition = flatten(possible_conditions[key])
            return condition, key


def get_consequence(doc, split_key):
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
#####################################################################################
#####################################################################################
# %% Low level

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
#####################################################################################
#####################################################################################

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
    if doc_pos_tags.count('CCONJ') < 2 and 'between' in [w.text for w in doc]:
        return [doc], binder
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


def get_rule_components(doc):
    rule = {'if':[], 'then':[], 'else':[]}
    else_part = []
    #  Step 1: get the cond, cons and else parts
    cond_cons = condition_consequence_extractor(doc)
    if cond_cons == 'No conditional statement in sentence':
        return get_lower_level_rule(doc)
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


def get_true_or_false(doc):
    for w in doc:
        if w.dep_ == 'neg':
            return 'False'
        elif 'no' in [w.text.lower() for w in doc]:
            return 'False'
    return 'True'


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


def get_comp_op(doc):
    doc_text = ' '.join([w.lemma_ for w in doc]).lower()
    less_than_syns = ['less than', 'few than', "do n't exceed ", "do 'nt surmount ", "don't pass ", 'young than', 'low than']
    less_equal_syns = ['less than or equal', 'less than or equal']
    greater_than_syns = ['great than', 'more than', 'exceed ', 'surmount ', ' pass ', 'be above', 'old than', 'long than']
    great_equal_syns = ['great than or equal', 'more than or equal']
    if ('between ' in doc_text or 'be within ' in doc_text or 'in ' in doc_text):
        return 'in'
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


def get_lower_level_rule(doc):
    if doc == []:
        return []
    comp_op = get_comp_op(doc)
    vars = []
    vals = []
    if comp_op == 'in':
        for word in doc:
            if word.pos_ in ['NUM']:
                vals.append(word)
            elif word.pos_ in ['NOUN', 'ADJ']:
                vars.append(word)
    else:
        for word in doc:
            if word.dep_ in ['nsubjpass','nsubj']:
                vars.append([w for w in word.subtree if w.pos_ in ['NOUN', 'ADJ', 'PROPN']])
            elif word.dep_ in ['dobj', 'pobj', 'attr', 'acomp']:
                vals.append([w for w in word.subtree if w.pos_ in ['NOUN', 'ADJ', 'NUM', 'PROPN', 'VERB']])
            # For action sentences
            elif word.dep_ in ['xcomp']:
                vars.append([w for w in word.subtree if w.pos_ in ['VERB']])

    if get_root(doc).pos_ == 'NOUN':
        vals.append(get_root(doc))
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
        comp_op = '='
        vals = get_true_or_false(doc)
    elif flatten(vars) == []:
        comp_op = '='
        vars = vals
        vals = get_true_or_false(doc)
    if vars != []:
        if 'spacy' in str(type(vars[0])):
            vars = remove_duplicate_chunks(vars)
            vars = order(vars)
    if vals != []:
        if 'spacy' in str(type(vals[0])):
            vals = remove_duplicate_chunks(vals)
            vals = order(vals)
    return vars, comp_op, vals


def order(doclist):
    output = []
    doclist_idx = [w.idx for w in doclist]
    doclist_idx.sort()
    doclist_idx.append('STOP')
    while doclist_idx != ['STOP']:
        for w in doclist:
            if w.idx == doclist_idx[0]:
                output.append(w)
        doclist_idx = doclist_idx[1:]
    return output



# ██████  ███████  ██████ ██ ███████ ██  ██████  ███    ██
# ██   ██ ██      ██      ██ ██      ██ ██    ██ ████   ██
# ██   ██ █████   ██      ██ ███████ ██ ██    ██ ██ ██  ██
# ██   ██ ██      ██      ██      ██ ██ ██    ██ ██  ██ ██
# ██████  ███████  ██████ ██ ███████ ██  ██████  ██   ████

# ██       ██████   ██████  ██  ██████
# ██      ██    ██ ██       ██ ██
# ██      ██    ██ ██   ███ ██ ██
# ██      ██    ██ ██    ██ ██ ██
# ███████  ██████   ██████  ██  ██████

# ███████ ██   ██ ████████ ██████   █████   ██████ ████████  ██████  ██████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██    ██ ██   ██
# █████     ███      ██    ██████  ███████ ██         ██    ██    ██ ██████
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██    ██ ██   ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██     ██████  ██   ██



def decision_logic_extractor(text):
    """
    This function takes as input a text fragment. This could be either a sentence or a paragraph with decision logic. The input value should be a string.
    """
    # Step 0: Make spacy document
    spacy_doc = sp(text)

    # Step 1: determine text type
    spacy_sentences = [sp(str(item)) for item in spacy_doc.sents]
    # Step 2: process each sentence individually
    highs = []
    lows = []
    for sentence in spacy_sentences:
        # Step 2.A: Extract high level information
        if condition_identifier(sentence):
            try:
                highs.append(condition_consequence_extractor(sentence))
            except:
                highs.append('error during extraction')
        # Step 2.B: Extract low level information
            try:
                lows.append(get_rule_components(sentence))
            except:
                lows.append('error during extraction')
        else:
            highs.append('None')
            lows.append('None')

    # Step 3: Make output list and return the result
    output_list = []
    i = 1
    for sentence, high, low in zip(spacy_sentences, highs, lows):
        output_list.append({'sentence {}'.format(i): sentence, 'high': high, 'low': low})
        i += 1
    return output_list
