from high_level_extractor import *
from low_level_extractor import *
from support_functions import *

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



sentences_spacy = get_spacy_lib(only_sentences)

# A discount of 4% and otherwise 9% is given when the order exceeds 10 units.
for sentence in sentences_spacy['Dataset_2']:
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
        print('LOW LEVEL')
        dictiona = get_full_dmn_rule(sentence)
        print('IF: ', dictiona['if'])
        print('THEN: ', dictiona['then'])
        print('ELSE: ', dictiona['else'])
        # print(dictiona['binder_if'], dictiona['binder_then'], dictiona['binder_else'])
    else:
        print(cond_cons)
        #print(get_lower_level_rule(sentence))
        print('')
