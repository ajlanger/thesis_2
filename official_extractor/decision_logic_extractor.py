from decision_logic_extractor_functions import *

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
