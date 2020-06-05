#############################################################
# USE OF INDIVIDUAL FUNCTIONS ###############################
#############################################################
# First import the decision logic extractor
from official_extractor.decision_logic_extractor_functions import *
# ---- Step 1: Make a spacy document of the sentence
doc = sp("If the day is rainy, sell waffles.")

# ---- Step 2a: Generate the high level logic
condition_consequence_extractor(doc)

# ---- Step 2b: Get the low level decision logic
get_rule_components(doc)

#############################################################
# USE OF DECISION LOGIC EXTRACTOR ###########################
#############################################################
from official_extractor.decision_logic_extractor import *

decision_logic = decision_logic_extractor(doc)
