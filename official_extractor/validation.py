# %% Import libraries
import decision_logic_extractor_functions
import importlib
importlib.reload(decision_logic_extractor_functions)
from decision_logic_extractor_functions import *

# %% ---- Functions for data exploration
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

# POS and DEP tag information
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
# Ex 1: If A then ACTION --> So only one object, one condition for object and one action
doc = sp("Tuscany sandwiches need to be made when the day is Thursday and the weather is sunny.")

# --------------------------------------------------------------------------------------------------
# Ex 2: If A and B then ACTION / If A or B then ACTION (SAME DEPENDENCY STRUCTURE FOR AND AND OR)

# I need to be able to search for synonyms of "no", "and" and "or"
doc = sp("If the person is between 22 and 29 years of age, and was involved in a car accident, insurance cost is 600 euros.")


# --------------------------------------------------------------------------------------------------
# %% consequence staments handler ------------------------------------------------------------------
# Then look at what condition the item/person should be in
doc = sp("In case that a person is between 19 and 21 years old and was not involved in a car accident, car insurance costs 500 euros.")

###################################################################################################
###################################################################################################

# All key elements in place, now last representation of the rules
doc1 = sp("If the customer stays for 2 days rent is 100 euro .")
doc2 = sp("If the customer stays for 2 days, rent is 100 euro .")

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

# --------------------------------------------------------------------------------------------------
# # Data processing part
# %% Extract all data from the text data and make list ---------------------------------------------
#filename = 'C:/Users/Arnaud/Google Drive/Master of AI/3. Thesis/thesis_2/raw_data.txt'
temp_list = get_texts('../textual_data/raw_data.txt')

# %% Make data dict --------------------------------------------------------------------------------
# Remove unnecesarry characters

only_sentences = get_only_sentences(temp_list)
# --------------------------------------------------------------------------------------------------
# %% Split into single sentences -------------------------------------------------------------------
# ....... with spaCy

test_data = pd.read_csv(r"../textual_data/test_data_csv_v2.csv", sep=';')
sentences_spacy = get_spacy_lib(only_sentences)

# A discount of 4% and otherwise 9% is given when the order exceeds 10 units.
for sentence in sentences_spacy['Dataset_1']:
    print('######################################################')
    print('----- NEXT SENTENCE -----')
    print(sentence)
    if 'doc' not in str(type(sentence)).lower():
        sentence = sp(str(sentence))
    #print(get_pos_tags_spacy(sentence))

    cond_cons = condition_consequence_extractor_v3(sentence)

    if cond_cons != 'No conditional statement in sentence' and cond_cons != 'No conditional statements could be extracted in spite of a condition being present.':
        print('HIGH LEVEL')
        #print(cond_cons)
        cond = sp(remove_conditional_words(make_string(cond_cons['condition'])))
        cons = sp(remove_consequence_words(make_string(cond_cons['consequence'])))
        print('--cond--')
        print(cond)
        print('--cons--')
        print(cons)
        print('')
        print('LOW LEVEL')
        dictiona = get_rule_components(sentence)
        print('IF: ', dictiona['if'])
        print('THEN: ', dictiona['then'])
        print('ELSE: ', dictiona['else'])
        # print(dictiona['binder_if'], dictiona['binder_then'], dictiona['binder_else'])
    else:
        print(cond_cons)
        print('')
