# First import the decision logic extractor
from official_extractor.decision_logic_extractor import *
# ---- Step 1: Make a spacy document of the sentence
doc = sp("If the day is rainy, sell waffles.")

# ---- Step 2a: Generate the high level logic
condition_consequence_extractor_v3(doc)

# ---- Step 2b: Get the low level decision logic
get_full_dmn_rule(doc)
