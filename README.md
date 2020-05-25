# Thesis on information retrieval

This work was written in order to succeed for the master's in AI at the University of Leuven. The module can be used to extract decision logic from a single sentence.

## Getting Started

In order to use the decision logic extractor, clone the repository to your local project directory. Make sure all required libraries are installed. The most important one is spaCy (https://spacy.io/) and the spaCy model "en_core_web_sm".

## Examples
Run the use_example.py to see how the extractor can be used.

```
# First import the decision logic extractor
from official_extractor.decision_logic_extractor import *

# ---- Step 1: Make a spaCy document of the sentence
doc = sp("If the day is rainy, sell waffles.")

# ---- Step 2a: Generate the high level logic
condition_consequence_extractor_v3(doc)

# ---- Step 2b: Get the low level decision logic
get_full_dmn_rule(doc)
```

The outputs are respectively:

```
OUTPUT STEP 2a
{'condition': [If, the, day, is, rainy], 'consequence': [,, sell, waffles]}

OUTPUT STEP 2b
{'if': [([day], '=', [rainy])],
 'then': [([waffles], '=', 'True')],
 'else': [[]]}
```

## Built With

* Python version 3.7.4
* spaCy version 2.2.4

## Authors

* **Arnaud Langeraert** 
