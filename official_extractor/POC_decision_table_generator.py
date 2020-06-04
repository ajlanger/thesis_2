from decision_logic_extractor import *

# Prepare validation data
temp_list = get_texts('../textual_data/raw_data.txt')
only_sentences = get_only_sentences(temp_list)
sentences_spacy = get_spacy_lib(only_sentences)

# Initialize lists for information extraction
if_vars = []
if_vals_1 = []
if_vals_2 = []
then_vars = []
then_vals = []

# Parse through the paragraph and append all information to appropriate arrays
for sentence in sentences_spacy['Dataset_2']:
    if 'doc' not in str(type(sentence)).lower():
        sentence = sp(str(sentence))
    cond_cons = condition_consequence_extractor(sentence)

    if cond_cons != 'No conditional statement in sentence' and cond_cons != 'No conditional statements could be extracted in spite of a condition being present.':
        dictiona = get_rule_components(sentence)
        if_vals_1.append((dictiona['if'][0][1], dictiona['if'][0][2]))
        if_vals_2.append((dictiona['if'][1][1], dictiona['if'][1][2]))
        if_vars.append(dictiona['if'][0][0])
        if_vars.append(dictiona['if'][1][0])
        for then_part in dictiona['then']:
            then_vars.append(then_part[0])
            then_vals.append((then_part[1], then_part[2]))

# Create a decision table using pandas
columns = ['INPUT VAR 1', 'INPUT VAR 2', 'OUTPUT VAR']
variables = [(str(make_string(if_vars[0]))), (str(make_string(if_vars[1]))), (str(make_string(then_vars[0])))]
if_vals_1.insert(0, variables[0]), if_vals_2.insert(0, variables[1]), then_vals.insert(0, variables[2])
pd_dict = {columns[0]: if_vals_1, columns[1]: if_vals_2, columns[2]: then_vals}
pd.DataFrame(pd_dict)
