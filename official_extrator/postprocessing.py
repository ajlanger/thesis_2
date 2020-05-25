# %% import libraries ------------------------------------------------------------------------------
from decision_logic_extractor import *

# %% Testing functions
#####################################################################################
#####################################################################################
# ████████ ███████ ███████ ████████ ██ ███    ██  ██████
#    ██    ██      ██         ██    ██ ████   ██ ██
#    ██    █████   ███████    ██    ██ ██ ██  ██ ██   ███
#    ██    ██           ██    ██    ██ ██  ██ ██ ██    ██
#    ██    ███████ ███████    ██    ██ ██   ████  ██████
#####################################################################################
#####################################################################################

def calculate_f_score(precision, recall):
    f_score = 2 * (precision*recall)/(precision+recall)
    return f_score


def get_precision_recall(actual_identified, identified, total):
    return round(actual_identified/identified, 3), round(actual_identified/total, 3)


def make_tuple(el):
    output_list = []
    for element in el:
        temp_tuple = []
        for e in element.split(','):
            if 'True' in e.replace('(', '').replace(')', '').strip():
                temp_tuple.append('True')
            elif 'False' in e.replace('(', '').replace(')', '').strip():
                temp_tuple.append('False')
            #elif e.replace('(', '').replace(')', '').strip()[0] in '0123456789':
            #    temp_tuple.append(float(e.replace('(', '').replace(')', '').strip()))
            elif e.replace('(', '').replace(')', '').strip()[0] == '[' and e.replace('(', '').replace(')', '').strip()[-1] == ']':
                interval = []
                for n in e.replace('(', '').replace(')', '').strip().split('..'):
                    interval.append(n)
                temp_tuple.append(interval)
            elif e.replace('(', '').replace(')', '').strip() in '=,!=,>,<,>=,=<,in':
                temp_tuple.append(e.replace('(', '').replace(')', '').strip())
            else:
                temp_tuple.append(e.replace('(', '').replace(')', '').strip().split(' '))
        output_list.append(tuple(temp_tuple))
    return output_list


def create_float(string):
    output = ''
    for letter in string:
        if letter in '1234567890':
            output += letter
    return float(output)


def format_column(column):
    # input : test_data['Low_else']
    all_lows = []
    for el in column:
        if el == 'None':
            all_lows.append(None)
        else:

            temp_list = []
            splitted = el.split('),')
            for element in splitted:
                if element[-1] == ')':
                    temp_list.append(element.strip())
                else:
                    temp_list.append((element + ')').strip())
            all_lows.append(make_tuple(temp_list))
    return all_lows


def count_rules(rule_list):
    count = 0
    for el in rule_list:
        if el != None:
            for i in el:
                count += 1
    return count


def extract_vars_vals_signs(row='', rowname='low_if', segment='if', extracted_low_level=''):
    extracted_vars = []
    extracted_vals = []
    extracted_signs = []
    desired_vars = []
    desired_vals = []
    desired_signs = []
    if extracted_low_level[segment][0] != []:
        for ifs in extracted_low_level[segment]:
            extracted_vars.append([word.text.lower() for word in flatten(ifs[0])])
            extracted_signs.append([flatten(ifs[1])])
            if type(ifs[2]) == list:
                extracted_vals.append([word.text.lower() for word in flatten(ifs[2])])
            else:
                extracted_vals.append([ifs[2]])
    for ifs in row[rowname]:
        desired_vars.append([word.lower() for word in flatten(ifs[0])])
        desired_signs.append([flatten(ifs[1])])
        if type(ifs[2]) == list:
            desired_vals.append([word.lower() for word in flatten(ifs[2])])
        else:
            desired_vals.append([ifs[2]])
    return extracted_vars, extracted_vals, extracted_signs, desired_vars, desired_vals, desired_signs

####################################################################################################
# %% Testing phase #################################################################################
# Import test data
test_data = pd.read_csv(r"../textual_data/test_data_csv_v2.csv", sep=';')

# Get logic in required format
desired_if_rules = format_column(test_data['Low_if'])
desired_then_rules = format_column(test_data['Low_then'])
desired_else_rules = format_column(test_data['Low_else'])

test_data['low_if'] = desired_if_rules
test_data['low_then'] = desired_then_rules
test_data['low_else'] = desired_else_rules

desired_total_ifs = count_rules(desired_if_rules)
desired_total_thens = count_rules(desired_then_rules)
desired_total_elses = count_rules(desired_else_rules)

# Actual identified
actual_identified_conditions, actual_identified_consequences, actual_identified_ifs_vars, actual_identified_thens_vars, actual_identified_elses_vars, actual_identified_ifs_vals, actual_identified_thens_vals, actual_identified_elses_vals, actual_identified_if_signs, actual_identified_then_signs, actual_identified_else_signs = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

# Identified concepts
identified_conditions, identified_consequences, identified_ifs_vars, identified_thens_vars, identified_elses_vars, identified_ifs_vals, identified_thens_vals, identified_elses_vals, identified_if_signs, identified_then_signs, identified_else_signs = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

for index, row in test_data.iterrows():
    # Automatic extraction
    extracted_cond_cons = condition_consequence_extractor_v3(sp(row['Sentences']))
    # Desired extractions
    desired_condition = row['Condition'].split(' ')
    desired_consequence = row['Consequence'].split(' ')
    # -----------------------------------------------------------------------------
    # Condition and consequences
    if extracted_cond_cons not in ['No conditional statements could be extracted in spite of a condition being present.', 'No conditional statement in sentence']:
        extracted_low_level = get_full_dmn_rule(sp(row['Sentences']))
        # -------------------------------------------------------------------------
        # identified conditions ---------------------------------------------------
        identified_conditions += 1
        extr_condition = [w.text for w in extracted_cond_cons['condition']]
        cond_identified = True
        for w in desired_condition:
            if w not in extr_condition:
                cond_identified = False
        if cond_identified:
            actual_identified_conditions += 1
        # -------------------------------------------------------------------------
        # identified consequences -------------------------------------------------
        identified_consequences += 1
        extr_consequence = [w.text for w in extracted_cond_cons['consequence']]
        cons_identified = True
        for w in desired_consequence:
            if w not in extr_consequence:
                cons_identified = False
        if cons_identified:
            actual_identified_consequences += 1
        # -------------------------------------------------------------------------
        # identified IFS ----------------------------------------------------------
        extracted_vars, extracted_vals, extracted_signs, desired_vars, desired_vals, desired_signs = extract_vars_vals_signs(row, 'low_if', 'if', extracted_low_level)
        if len(extracted_vars) == len(desired_vars):
            for i in range(len(extracted_vars)):
                # IF Vars
                vars_identified = True
                identified_ifs_vars += 1
                for element in desired_vars[i]:
                    if element not in extracted_vars[i]:
                        vars_identified = False
                if vars_identified:
                    actual_identified_ifs_vars += 1
                # IF Vals
                vals_identified = True
                identified_ifs_vals += 1
                for element in desired_vals[i]:
                    if element not in extracted_vals[i]:
                        vals_identified = False
                if vals_identified:
                    actual_identified_ifs_vals += 1
                # IF signs
                signs_identified = True
                identified_if_signs += 1
                for element in desired_signs[i]:
                    if element not in extracted_signs[i]:
                        signs_identified = False
                        mistakes += 1
                if signs_identified:
                    actual_identified_if_signs += 1
        else:
            all_vars_extracted, all_vals_extracted, all_signs_extracted, all_vars_desired, all_vals_desired, all_signs_desired = flatten(extracted_vars), flatten(extracted_vals), flatten(extracted_signs), flatten(desired_vars), flatten(desired_vals), flatten(desired_signs)
            # IF Vars
            vars_identified = True
            identified_ifs_vars += 1
            for element in all_vars_desired:
                if element not in all_vars_extracted:
                    vars_identified = False
            if vars_identified:
                actual_identified_ifs_vars += 1
            # IF Vals
            vals_identified = True
            identified_ifs_vals += 1
            for element in all_vals_desired:
                if element not in all_vals_extracted:
                    vals_identified = False
            if vals_identified:
                actual_identified_ifs_vals += 1
            # IF signs
            signs_identified = True
            identified_if_signs += 1
            for element in all_signs_desired:
                if element not in all_signs_extracted:
                    signs_identified = False
            if signs_identified:
                actual_identified_if_signs += 1

        # -------------------------------------------------------------------------
        # identified THENS ----------------------------------------------------------
        extracted_vars, extracted_vals, extracted_signs, desired_vars, desired_vals, desired_signs = extract_vars_vals_signs(row, 'low_then', 'then', extracted_low_level)
        if len(extracted_vars) == len(desired_vars):
            for i in range(len(extracted_vars)):
                # Then Vars
                vars_identified = True
                identified_thens_vars += 1
                for element in desired_vars[i]:
                    if element not in extracted_vars[i]:
                        vars_identified = False
                if vars_identified:
                    actual_identified_thens_vars += 1
                # Then Vals
                vals_identified = True
                identified_thens_vals += 1
                for element in desired_vals[i]:
                    if element not in extracted_vals[i]:
                        vals_identified = False
                if vals_identified:
                    actual_identified_thens_vals += 1
                # Then signs
                signs_identified = True
                identified_then_signs += 1
                for element in desired_signs[i]:
                    if element not in extracted_signs[i]:
                        signs_identified = False
                if signs_identified:
                    actual_identified_then_signs += 1
        else:
            all_vars_extracted, all_vals_extracted, all_signs_extracted, all_vars_desired, all_vals_desired, all_signs_desired = flatten(extracted_vars), flatten(extracted_vals), flatten(extracted_signs), flatten(desired_vars), flatten(desired_vals), flatten(desired_signs)
            # Then Vars
            vars_identified = True
            identified_thens_vars += 1
            for element in all_vars_desired:
                if element not in all_vars_extracted:
                    vars_identified = False
            if vars_identified:
                actual_identified_thens_vars += 1
            # Then Vals
            vals_identified = True
            identified_thens_vals += 1
            for element in all_vals_desired:
                if element not in all_vals_extracted:
                    vals_identified = False
            if vals_identified:
                actual_identified_thens_vals += 1
            # Then signs
            signs_identified = True
            identified_then_signs += 1
            for element in all_signs_desired:
                if element not in all_signs_extracted:
                    signs_identified = False
            if signs_identified:
                actual_identified_then_signs += 1

        # -------------------------------------------------------------------------
        # identified ELSES ----------------------------------------------------------
        if row['low_else'] != None:
            extracted_vars, extracted_vals, extracted_signs, desired_vars, desired_vals, desired_signs = extract_vars_vals_signs(row, 'low_else', 'else', extracted_low_level)

            if len(extracted_vars) == len(desired_vars):
                for i in range(len(extracted_vars)):
                    # Else Vars
                    vars_identified = True
                    identified_elses_vars += 1
                    for element in desired_vars[i]:
                        if element not in extracted_vars[i]:
                            vars_identified = False
                    if vars_identified:
                        actual_identified_elses_vars += 1
                    # Else Vals
                    vals_identified = True
                    identified_elses_vals += 1
                    for element in desired_vals[i]:
                        if element not in extracted_vals[i]:
                            vals_identified = False
                    if vals_identified:
                        actual_identified_elses_vals += 1
                    # Else signs
                    signs_identified = True
                    identified_else_signs += 1
                    for element in desired_signs[i]:
                        if element not in extracted_signs[i]:
                            signs_identified = False
                    if signs_identified:
                        actual_identified_else_signs += 1
            else:
                all_vars_extracted, all_vals_extracted, all_signs_extracted, all_vars_desired, all_vals_desired, all_signs_desired = flatten(extracted_vars), flatten(extracted_vals), flatten(extracted_signs), flatten(desired_vars), flatten(desired_vals), flatten(desired_signs)
                # Else Vars
                vars_identified = True
                identified_elses_vars += 1
                for element in all_vars_desired:
                    if element not in all_vars_extracted:
                        vars_identified = False
                if vars_identified:
                    actual_identified_elses_vars += 1
                # Else Vals
                vals_identified = True
                identified_elses_vals += 1
                for element in all_vals_desired:
                    if element not in all_vals_extracted:
                        vals_identified = False
                if vals_identified:
                    actual_identified_elses_vals += 1
                # Else signs
                signs_identified = True
                identified_else_signs += 1
                for element in all_signs_desired:
                    if element not in all_signs_extracted:
                        signs_identified = False
                if signs_identified:
                    actual_identified_else_signs += 1
    else:
        print("##############################################")
        print("##### THIS ONE COULD NOT BE EXTRACTED #####")
        print('Sentence: ', row['Sentences'])
        print("##############################################")

get_dep_parse(sp("If it 's Christmas, sell turkey, else sell chicken and potatoes."))

# %%
#################################################################################################
#################################################################################################
# ██████  ██████  ███████  ██████ ██ ███████ ██  ██████  ███    ██
# ██   ██ ██   ██ ██      ██      ██ ██      ██ ██    ██ ████   ██
# ██████  ██████  █████   ██      ██ ███████ ██ ██    ██ ██ ██  ██
# ██      ██   ██ ██      ██      ██      ██ ██ ██    ██ ██  ██ ██
# ██      ██   ██ ███████  ██████ ██ ███████ ██  ██████  ██   ████

# ██████  ███████  ██████  █████  ██      ██
# ██   ██ ██      ██      ██   ██ ██      ██
# ██████  █████   ██      ███████ ██      ██
# ██   ██ ██      ██      ██   ██ ██      ██
# ██   ██ ███████  ██████ ██   ██ ███████ ███████

# ██████  ███████ ███████ ██    ██ ██   ████████ ███████
# ██   ██ ██      ██      ██    ██ ██      ██    ██
# ██████  █████   ███████ ██    ██ ██      ██    ███████
# ██   ██ ██           ██ ██    ██ ██      ██         ██
# ██   ██ ███████ ███████  ██████  ███████ ██    ███████
#################################################################################################
#################################################################################################


#################################################################################################
# HIGH LEVEL RESULTS ############################################################################
#################################################################################################
# Calculation of precision - cond --> actual_identified_conditions/identified_conditions
# Calculation of recall - cond --> actual_identified_conditions/total_real_conditions
cond_precision, cond_recall = get_precision_recall(actual_identified_conditions, identified_conditions, 92)

print('cond_precision: ', cond_precision)
print('cond_recall: ', cond_recall)

cond_f_score = calculate_f_score(cond_precision, cond_recall)
cond_f_score

# Calculation of precision - cons --> actual_identified_consequences/identified_consequences
# Calculation of recall - cons --> actual_identified_consequences/total_real_consequences
cons_precision, cons_recall = get_precision_recall(actual_identified_consequences, identified_consequences, 92)

print('cons_precision: ', cons_precision)
print('cons_recall: ', cons_recall)

cons_f_score = calculate_f_score(cons_precision, cons_recall)
cons_f_score

#################################################################################################
# LOW LEVEL RESULTS #############################################################################
#################################################################################################
# If vars, vals & signs -------------------------------------------------------------
if_var_precision, if_var_recall = get_precision_recall(actual_identified_ifs_vars, identified_ifs_vars, desired_total_ifs)
print('if_variables_precision: ', if_var_precision)
print('if_variables_recall: ', if_var_recall)
if_var_f_score = calculate_f_score(if_var_precision, if_var_recall)
if_var_f_score

if_val_precision, if_val_recall = get_precision_recall(actual_identified_ifs_vals, identified_ifs_vals, desired_total_ifs)
print('if_values_precision: ', if_val_precision)
print('if_values_recall: ', if_val_recall)
if_val_f_score = calculate_f_score(if_val_precision, if_val_recall)
if_val_f_score

if_sign_precision, if_sign_recall = get_precision_recall(actual_identified_if_signs, identified_if_signs, desired_total_ifs)
print('if_sign_precision: ', if_sign_precision)
print('if_sign_recall: ', if_sign_recall)
if_sign_f_score = calculate_f_score(if_sign_precision, if_sign_recall)
if_sign_f_score

if_f_score = (if_var_f_score + if_val_f_score + if_sign_f_score)/3

# THEN vars, vals & signs -------------------------------------------------------------
then_var_precision, then_var_recall = get_precision_recall(actual_identified_thens_vars, identified_thens_vars, desired_total_thens)
print('then_variables_precision: ', then_var_precision)
print('then_variables_recall: ', then_var_recall)
then_var_f_score = calculate_f_score(then_var_precision, then_var_recall)
then_var_f_score

then_val_precision, then_val_recall = get_precision_recall(actual_identified_thens_vals, identified_thens_vals, desired_total_thens)
print('then_values_precision: ', then_val_precision)
print('then_values_recall: ', then_val_recall)
then_val_f_score = calculate_f_score(then_val_precision, then_val_recall)
then_val_f_score

then_sign_precision, then_sign_recall = get_precision_recall(actual_identified_then_signs, identified_then_signs, desired_total_thens)
print('then_sign_precision: ', then_sign_precision)
print('then_sign_recall: ', then_sign_recall)
then_sign_f_score = calculate_f_score(then_sign_precision, then_sign_recall)
then_sign_f_score

then_f_score = (then_var_f_score + then_val_f_score + then_sign_f_score)/3

# ELSE vars, vals & signs -------------------------------------------------------------
else_var_precision, else_var_recall = get_precision_recall(actual_identified_elses_vars, identified_elses_vars, desired_total_elses)
print('else_variables_precision: ', else_var_precision)
print('else_variables_recall: ', else_var_recall)
else_var_f_score = calculate_f_score(else_var_precision, else_var_recall)
else_var_f_score

else_val_precision, else_val_recall = get_precision_recall(actual_identified_elses_vals, identified_elses_vals, desired_total_elses)
print('else_values_precision: ', else_val_precision)
print('else_values_recall: ', else_val_recall)
else_val_f_score = calculate_f_score(else_val_precision, else_val_recall)
else_val_f_score

else_sign_precision, else_sign_recall = get_precision_recall(actual_identified_else_signs, identified_else_signs, desired_total_elses)
print('else_sign_precision: ', else_sign_precision)
print('else_sign_recall: ', else_sign_recall)
else_sign_f_score = calculate_f_score(else_sign_precision, else_sign_recall)
else_sign_f_score

else_f_score = (else_var_f_score + else_val_f_score + else_sign_f_score)/3

##########################################################################################
# OVERALL F-SCORE
high_level_extractor_f_score = (cond_f_score+cons_f_score)/2
high_level_extractor_f_score

low_level_extractor_f_score = (if_f_score+then_f_score+else_f_score)/3
low_level_extractor_f_score

overall_f_score = (high_level_extractor_f_score + low_level_extractor_f_score)/2
overall_f_score