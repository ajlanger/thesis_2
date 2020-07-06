# Thesis on information retrieval

This work was written in order to succeed for the master's in AI at the University of Leuven. The module can be used to extract decision logic from a single sentence.

## Getting Started

In order to use the decision logic extractor, clone the repository to your local project directory. Make sure all required libraries are installed. The most important one is spaCy (https://spacy.io/) and the spaCy model "en_core_web_sm".

## Examples
Run the use_example.py to see how the extractor can be used.

```
#############################################################
# USE OF INDIVIDUAL FUNCTIONS ###############################
#############################################################

# First import the decision logic extractor
from official_extractor.decision_logic_extractor_functions import *

# ---- Step 1: Make a spaCy document of the sentence
doc = "If the day is rainy, sell waffles."

# ---- Step 2a: Generate the high level logic
condition_consequence_extractor(sp(doc))

# ---- Step 2b: Get the low level decision logic
get_rule_components(sp(doc))

#############################################################
# USE OF DECISION LOGIC EXTRACTOR ###########################
#############################################################
from official_extractor.decision_logic_extractor_functions import *

decision_logic = decision_logic_extractor(doc)
```

The outputs are respectively:

```
OUTPUT STEP 2a
{'condition': [If, the, day, is, rainy], 'consequence': [,, sell, waffles]}

OUTPUT STEP 2b
{'if': [([day], '=', [rainy])],
 'then': [([waffles], '=', 'True')],
 'else': [[]]}

 OUTPUT DECISION LOGIC EXTRACTOR
 [{'sentence 1': If the day is rainy, sell waffles.,

  'high': {'condition': [If, the, day, is, rainy],
           'consequence': [,, sell, waffles, .]},

  'low': {'if': [([day], '=', [rainy])],
          'then': [([waffles], '=', 'True')],
          'else': [[]]}}]

```

## Built With

* Python version 3.7.4
* spaCy version 2.2.4

## Authors

* **Arnaud Langeraert**
